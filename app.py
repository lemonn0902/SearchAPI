
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.genai.types import Part
from google.cloud import storage, firestore
import os
import traceback
import uuid
import hashlib
from datetime import datetime
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# # Start ngrok
# ngrok.set_auth_token("2yixWcyavcFl6DGPUMEOQZ2QKiJ_3Jm9iKnYb1svHPJg9Ltz7")
# print("🔗 Public URL:", public_url)

# Add caching variables at the top after imports
pdf_cache = {}
cache_timestamp = None
CACHE_DURATION = 1800  # 30 minutes in seconds
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
PROJECT_ID = "orbital-wharf-441212-g2"  # Add this line
MODEL_ID = "gemini-2.0-flash-001"
BUCKET_NAME = "spikedbucket1"  # Replace with your bucket name

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)
firestore_client = firestore.Client(project=PROJECT_ID)

def check_file_exists_in_bucket(filename, file_size):
    """Check if a file with the same name and size already exists in bucket"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs()

        for blob in blobs:
            # Extract original filename from stored filename (it's at the end after the last underscore)
            stored_name_parts = blob.name.split('_')
            if len(stored_name_parts) >= 4:
                # Format is: pdfs/{timestamp}_{uuid}_{original_name}
                original_name = '_'.join(stored_name_parts[3:])  # Join in case original name had underscores

                if original_name == filename and blob.size == file_size:
                    return {
                        "exists": True,
                        "gcs_path": f"gs://{BUCKET_NAME}/{blob.name}",
                        "public_url": f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}",
                        "size_bytes": blob.size,
                        "existing_filename": blob.name
                    }

        return {"exists": False}
    except Exception as e:
        print(f"🔥 Error checking file existence: {str(e)}")
        return {"exists": False}

def store_pdfs_in_gcs(pdf_files):
    """Store PDFs in Google Cloud Storage and return their URLs, skip duplicates"""
    stored_files = []

    try:
        for file in pdf_files:
            if file.filename:
                file.seek(0)  # Reset file pointer to get size
                file_content = file.read()
                file_size = len(file_content)

                # Check if file already exists (by name and size)
                existing_file = check_file_exists_in_bucket(file.filename, file_size)

                if existing_file["exists"]:
                    print(f"📄 File {file.filename} already exists (size: {file_size} bytes), skipping upload")
                    stored_files.append({
                        "original_name": file.filename,
                        "gcs_path": existing_file["gcs_path"],
                        "public_url": existing_file["public_url"],
                        "size_bytes": existing_file["size_bytes"],
                        "upload_time": "existing_file",
                        "status": "skipped_duplicate",
                        "existing_filename": existing_file["existing_filename"]
                    })
                else:
                    # Upload new file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"pdfs/{timestamp}_{unique_id}_{file.filename}"

                    # Upload to GCS
                    bucket = storage_client.bucket(BUCKET_NAME)
                    blob = bucket.blob(filename)

                    file.seek(0)  # Reset file pointer
                    blob.upload_from_file(file, content_type='application/pdf')

                    print(f"📁 Uploaded new file {file.filename} (size: {file_size} bytes)")
                    stored_files.append({
                        "original_name": file.filename,
                        "gcs_path": f"gs://{BUCKET_NAME}/{filename}",
                        "public_url": f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}",
                        "size_bytes": blob.size,
                        "upload_time": datetime.now().isoformat(),
                        "status": "uploaded_new"
                    })

                    # Invalidate cache since we added a new file
                    invalidate_cache()

        return stored_files

    except Exception as e:
        print(f"🔥 Error storing files in GCS: {str(e)}")
        raise Exception(f"Storage error: {str(e)}")

def get_cached_answer(question):
    """Check if we have a cached answer for this exact question"""
    try:
        # Normalize the question (lowercase, strip whitespace)
        normalized_question = question.lower().strip()

        # Query Firestore for existing answers
        query = firestore_client.collection('pdf_queries').where('question_normalized', '==', normalized_question).limit(1)
        docs = list(query.stream())

        if docs:
            doc_data = docs[0].to_dict()
            print(f"💨 Found cached answer for question: {question[:50]}...")
            return {
                "answer": doc_data.get('answer'),
                "cached": True,
                "original_timestamp": doc_data.get('timestamp'),
                "doc_id": docs[0].id
            }

        return None
    except Exception as e:
        print(f"🔥 Error checking cached answer: {str(e)}")
        return None

def invalidate_cache():
    """Clear only the answer cache from Firestore, keep PDF files intact"""
    try:
        # Clear answer cache from Firestore
        batch = firestore_client.batch()
        docs = firestore_client.collection('pdf_queries').stream()

        deleted_count = 0
        for doc in docs:
            batch.delete(doc.reference)
            deleted_count += 1

            # Firestore batch operations have a limit of 500 operations
            if deleted_count % 500 == 0:
                batch.commit()
                batch = firestore_client.batch()

        # Commit any remaining operations
        if deleted_count % 500 != 0:
            batch.commit()

        print(f"🗑️ Cleared {deleted_count} cached answers from Firestore")
        print("✅ Answer cache cleared successfully (PDFs preserved)")

        return {"cleared_answers": deleted_count}

    except Exception as e:
        print(f"🔥 Error clearing answer cache: {str(e)}")
        raise Exception(f"Answer cache clearing error: {str(e)}")

def store_metadata_in_firestore(question, stored_files, answer):
    """Store processing metadata in Firestore"""
    try:
        # Normalize the question for caching
        normalized_question = question.lower().strip()

        doc_ref = firestore_client.collection('pdf_queries').document()
        doc_ref.set({
            'question': question,
            'question_normalized': normalized_question,  # Add this for caching
            'files': stored_files,
            'answer': answer,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'processed': True,
            'file_count': len(stored_files),
            'new_uploads': len([f for f in stored_files if f.get('status') == 'uploaded_new']),
            'skipped_duplicates': len([f for f in stored_files if f.get('status') == 'skipped_duplicate'])
        })
        print(f"💾 Stored answer cache for question: {question[:50]}...")
        return doc_ref.id
    except Exception as e:
        print(f"🔥 Error storing metadata in Firestore: {str(e)}")
        return None

def get_all_pdfs_from_bucket():
    """Get all PDF files from GCS bucket with caching"""
    global pdf_cache, cache_timestamp

    try:
        current_time = datetime.now().timestamp()

        # Check if cache is valid (less than 5 minutes old)
        if cache_timestamp and (current_time - cache_timestamp) < CACHE_DURATION and pdf_cache:
            print("📋 Using cached PDFs")
            return list(pdf_cache.values())

        print("📁 Loading PDFs from bucket...")
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs()

        new_cache = {}
        for blob in blobs:
            if blob.name.lower().endswith('.pdf'):
                # Only download if not in cache or cache is old
                if blob.name not in pdf_cache:
                    print(f"📄 Loading {blob.name}")
                    pdf_bytes = blob.download_as_bytes()
                    if pdf_bytes:
                        new_cache[blob.name] = pdf_bytes
                else:
                    # Reuse cached version
                    new_cache[blob.name] = pdf_cache[blob.name]

        pdf_cache = new_cache
        cache_timestamp = current_time
        print(f"📚 Loaded {len(pdf_cache)} PDFs into cache")

        return list(pdf_cache.values())
    except Exception as e:
        print(f"🔥 Error getting PDFs from bucket: {str(e)}")
        return []

def ask_with_multiple_pdfs(tokens, prompt, question, pdf_files=None):
    try:
        parts = []

        # If pdf_files provided, use them
        if pdf_files:
            for file in pdf_files:
                file.seek(0)  # Reset file pointer
                file_bytes = file.read()
                if not file_bytes:
                    continue
                parts.append(Part(inline_data={"data": file_bytes, "mime_type": "application/pdf"}))
        else:
            # If no pdf_files provided, get all PDFs from bucket (cached)
            bucket_pdfs = get_all_pdfs_from_bucket()
            for pdf_bytes in bucket_pdfs:
                parts.append(Part(inline_data={"data": pdf_bytes, "mime_type": "application/pdf"}))

        parts.append(question)
        parts.append(prompt)


        response = client.models.generate_content(
            model=MODEL_ID,
            contents=parts,
            config=GenerateContentConfig(
                temperature=0.2,  # Reduced for faster processing
                top_p=0.9,        # Slightly higher for efficiency
                top_k=10,         # Reduced for speed
                candidate_count=1,
                seed=5,
                max_output_tokens=tokens,  # Reduced from 1000
                stop_sequences=["STOP!"],
                presence_penalty=0.0,
                frequency_penalty=0.0,
                response_logprobs=False,
            ),
        )
        return response.text

    except Exception as e:
        print("🔥 Error during Gemini processing:", traceback.format_exc())
        raise Exception(f"Gemini model error: {str(e)}")

@app.route('/askfaster', methods=['POST'])
def askfaster():
    try:
        question = request.form.get('question')
        pdf_files = request.files.getlist('pdfs')

        if not question:
            return jsonify({"error": "Missing question"}), 400

        # First check if we have a cached answer for this question
        cached_result = get_cached_answer(question)
        if cached_result:
            return jsonify({
                "answer": cached_result["answer"],
                "cached": True,
                "original_timestamp": cached_result["original_timestamp"],
                "cached_doc_id": cached_result["doc_id"],
                "processing_time": "instant"
            })

        stored_files = []

        # If PDFs are provided, validate and store them
        if pdf_files and len(pdf_files) > 0:
            # Validate file types
            for file in pdf_files:
                if not file.filename.lower().endswith('.pdf'):
                    return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

            # Store PDFs in GCS (will skip duplicates)
            print("📁 Checking and storing PDFs in Google Cloud Storage...")
            stored_files = store_pdfs_in_gcs(pdf_files)

        # Always process with ALL PDFs from bucket (regardless of whether new files were uploaded)
        print("🤖 Processing with Gemini using ALL PDFs from bucket...")
        shorter_prompt= """Generate a clear, detailed answer to the user's question using insights from the provided PDFs. Ensure it reflects deep understanding and context.

Then include:

BANT Follow-up Question
Give one natural, client-ready question based on BANT (choose from: Budget, Authority, Need, or Timeline).

MEDDPICC Questions
Refer to meddpicc.pdf. Pick three distinct MEDDPICC components and for each, write one full, client-facing question a sales rep could ask. Make them specific and drawn from or inspired by the PDF.

Format:

[Component]: [Client-ready question]

[Component]: [Client-ready question]

[Component]: [Client-ready question]"""
        answer = ask_with_multiple_pdfs(500, shorter_prompt, question)

        # Store metadata in Firestore (this creates the cache for future use)
        print("📊 Storing metadata in Firestore...")
        doc_id = store_metadata_in_firestore(question, stored_files, answer)

        return jsonify({
            "answer": answer,
            "cached": False,
            "stored_files": stored_files,
            "metadata_doc_id": doc_id,
            "storage_info": {
                "bucket": BUCKET_NAME,
                "file_count": len(stored_files),
                "new_uploads": len([f for f in stored_files if f.get('status') == 'uploaded_new']),
                "skipped_duplicates": len([f for f in stored_files if f.get('status') == 'skipped_duplicate']),
                "total_size_mb": sum(f.get('size_bytes', 0) for f in stored_files) / (1024 * 1024) if stored_files else 0
            }
        })

    except Exception as e:
        print("🔥 Error in /ask route:", traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.form.get('question')
        pdf_files = request.files.getlist('pdfs')

        if not question:
            return jsonify({"error": "Missing question"}), 400

        # First check if we have a cached answer for this question
        cached_result = get_cached_answer(question)
        if cached_result:
            return jsonify({
                "answer": cached_result["answer"],
                "cached": True,
                "original_timestamp": cached_result["original_timestamp"],
                "cached_doc_id": cached_result["doc_id"],
                "processing_time": "instant"
            })

        stored_files = []

        # If PDFs are provided, validate and store them
        if pdf_files and len(pdf_files) > 0:
            # Validate file types
            for file in pdf_files:
                if not file.filename.lower().endswith('.pdf'):
                    return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

            # Store PDFs in GCS (will skip duplicates)
            print("📁 Checking and storing PDFs in Google Cloud Storage...")
            stored_files = store_pdfs_in_gcs(pdf_files)
        prompt =prompt= """
Generate a comprehensive, well-structured response that directly addresses the user's question, drawing accurately and insightfully from the content of the provided PDFs. Ensure the response is detailed, contextual, and demonstrates a deep understanding of the material.
After your main response, please include the following two sections:

---
🔍 BANT Follow-up Questions
Provide one good related follow-up questions using the BANT framework:

- Budget: Can they afford it?
- Authority: Are they the decision maker?
- Need: What specific problem are they solving?
- Timeline: When do they plan to move forward?

---

📌 MEDDPICC Suggested Questions
Use the file `meddpicc.pdf` from the bucket to extract actual sales qualification questions that align with the **MEDDPICC** framework. Select **three distinct components** from the framework and for each one, provide **one natural, client-facing question** that a sales rep can ask. Ensure these are **not just labels**, but fully phrased, relevant questions drawn from or inspired by the content in `meddpicc.pdf`.
MEDDPICC Framework Components:
- **Metrics**: What measurable outcomes matter to them?
- **Economic Buyer**: Who controls the budget and final decision?
- **Decision Criteria**: What factors will they use to evaluate solutions?
- **Decision Process**: How do they typically make purchasing decisions?
- **Paper Process**: What's their procurement/approval process?
- **Identify Pain**: What specific problems are they trying to solve?
- **Champion**: Who internally supports this initiative?
- **Competition**: What alternatives are they considering?

Format your response as:
1. [Component]: [Full question a sales rep would ask]
2. [Component]: [Full question a sales rep would ask]
3. [Component]: [Full question a sales rep would ask]

Example format:
1. Metrics: What measurable outcomes are you expecting from this solution?
2. Champion: Who on your team would be most motivated to drive this change internally?

Do NOT just list component names - provide the actual questions.
Make sure all questions are tailored to the context and phrased naturally for client conversations.
"""
        # Always process with ALL PDFs from bucket (regardless of whether new files were uploaded)
        print("🤖 Processing with Gemini using ALL PDFs from bucket...")
        answer = ask_with_multiple_pdfs(1000, prompt, question )

        # Store metadata in Firestore (this creates the cache for future use)
        print("📊 Storing metadata in Firestore...")
        doc_id = store_metadata_in_firestore(question, stored_files, answer)

        return jsonify({
            "answer": answer,
            "cached": False,
            "stored_files": stored_files,
            "metadata_doc_id": doc_id,
            "storage_info": {
                "bucket": BUCKET_NAME,
                "file_count": len(stored_files),
                "new_uploads": len([f for f in stored_files if f.get('status') == 'uploaded_new']),
                "skipped_duplicates": len([f for f in stored_files if f.get('status') == 'skipped_duplicate']),
                "total_size_mb": sum(f.get('size_bytes', 0) for f in stored_files) / (1024 * 1024) if stored_files else 0
            }
        })

    except Exception as e:
        print("🔥 Error in /ask route:", traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/files', methods=['GET'])
def list_stored_files():
    """List all stored files from Firestore"""
    try:
        docs = firestore_client.collection('pdf_queries').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()

        files_history = []
        for doc in docs:
            data = doc.to_dict()
            files_history.append({
                'id': doc.id,
                'question': data.get('question'),
                'file_count': data.get('file_count', 0),
                'new_uploads': data.get('new_uploads', 0),
                'skipped_duplicates': data.get('skipped_duplicates', 0),
                'timestamp': data.get('timestamp'),
                'files': data.get('files', [])
            })

        return jsonify({"files_history": files_history})

    except Exception as e:
        print("🔥 Error listing files:", traceback.format_exc())
        return jsonify({"error": f"Error retrieving files: {str(e)}"}), 500

# Add a route to clear cache for testing
# Updated route to clear answer cache only
@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear only the answer cache for testing purposes"""
    try:
        # Count Firestore docs before clearing
        firestore_count_before = len(list(firestore_client.collection('pdf_queries').stream()))

        # Clear only answer cache
        cache_stats = invalidate_cache()

        return jsonify({
            "message": "Answer cache cleared successfully",
            "cleared": {
                "cached_answers": firestore_count_before
            },
            "preserved": [
                "PDF files in Google Cloud Storage",
                "PDF file cache in memory"
            ],
            "note": "Only cached answers were cleared. PDFs remain available for future queries."
        })

    except Exception as e:
        return jsonify({
            "error": f"Error clearing answer cache: {str(e)}",
            "note": "PDF files were not affected"
        }), 500

if __name__ == '__main__':
    app.run()

