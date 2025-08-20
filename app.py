import os
import random
import time
from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
import json
from flask_cors import CORS
from openai import OpenAI
import requests
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)

# Environment setup
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=openai_api_key)

# WebShare proxy configuration from your dashboard
WEBSHARE_PROXIES = [
    {"ip": "23.95.150.145", "port": 6114, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
    {"ip": "198.23.239.134", "port": 6540, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
    {"ip": "45.38.107.97", "port": 6014, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
    {"ip": "107.172.163.27", "port": 6543, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
    {"ip": "64.137.96.74", "port": 6641, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
    {"ip": "45.43.186.39", "port": 6257, "username": "dofmcoom", "password": "k8gjbcts7ekn"},
]

def get_random_proxy():
    """Get a random working proxy from WebShare"""
    proxy = random.choice(WEBSHARE_PROXIES)
    proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['ip']}:{proxy['port']}"
    
    return {
        'http': proxy_url,
        'https': proxy_url
    }

def get_video_id(url):
    patterns = [
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?v=)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def chunk_text(text, max_chunk_size=4000):
    """Improved text chunking with better word boundary handling"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_size + word_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_len
        else:
            current_chunk.append(word)
            current_size += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_transcript_with_webshare_proxy(video_id, max_retries=3):
    """
    Fetch transcript using WebShare proxies with fallback
    """
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Trying with WebShare proxy")
            
            # Get a random proxy
            proxy_dict = get_random_proxy()
            print(f"Using proxy: {proxy_dict['http'].split('@')[1]}")  # Log proxy IP (without credentials)
            
            # Method 1: Direct proxy approach using requests
            try:
                # Build the transcript URL manually
                transcript_url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json3"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
                }
                
                response = requests.get(
                    transcript_url, 
                    headers=headers, 
                    proxies=proxy_dict, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    content = response.text
                    if content and len(content) > 100:
                        # Parse YouTube's JSON3 format
                        transcript_data = json.loads(content)
                        if 'events' in transcript_data:
                            transcript = []
                            for event in transcript_data['events']:
                                if 'segs' in event:
                                    text_parts = []
                                    for seg in event['segs']:
                                        if 'utf8' in seg:
                                            text_parts.append(seg['utf8'])
                                    if text_parts:
                                        transcript.append({
                                            'text': ''.join(text_parts),
                                            'start': event.get('tStartMs', 0) / 1000.0,
                                            'duration': event.get('dDurationMs', 0) / 1000.0
                                        })
                            
                            if transcript:
                                print(f"Successfully fetched transcript via direct API with proxy")
                                return transcript
                
            except Exception as e:
                print(f"Direct API method failed: {str(e)}")
            
            # Method 2: YouTube Transcript API with proxy (if available)
            try:
                # Note: youtube-transcript-api doesn't directly support requests-style proxies
                # This is a workaround by monkey-patching
                import youtube_transcript_api._api
                original_get = requests.Session.get
                
                def proxied_get(self, *args, **kwargs):
                    kwargs['proxies'] = proxy_dict
                    return original_get(self, *args, **kwargs)
                
                # Temporarily patch the requests
                requests.Session.get = proxied_get
                
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
                
                # Restore original method
                requests.Session.get = original_get
                
                print(f"Successfully fetched transcript via YouTube Transcript API with proxy")
                return transcript
                
            except Exception as e:
                print(f"YouTube Transcript API with proxy failed: {str(e)}")
                # Restore original method just in case
                requests.Session.get = original_get
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                
    # Final fallback: Try without proxy
    try:
        print("Trying final fallback without proxy")
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        print("Fallback without proxy succeeded")
        return transcript
    except Exception as e:
        raise Exception(f"All methods failed including non-proxy fallback. Last error: {str(e)}")

def analyze_content_structure(full_text: str) -> Dict[str, Any]:
    """Analyze the transcript to identify key topics, concepts, and structure"""
    
    # Truncate text for analysis to avoid token limits
    analysis_text = full_text[:2500] if len(full_text) > 2500 else full_text
    
    analysis_prompt = f"""Analyze this video transcript and provide a structured analysis in JSON format:

    Transcript: {analysis_text}

    Return ONLY valid JSON with this structure:
    {{
        "main_topics": ["topic1", "topic2", "topic3"],
        "key_concepts": ["concept1", "concept2", "concept3"],
        "content_type": "educational|tutorial|interview|presentation|discussion",
        "difficulty_level": "beginner|intermediate|advanced",
        "subject_area": "technology|science|business|arts|health|finance|entertainment|other",
        "important_facts": ["fact1", "fact2", "fact3"],
        "actionable_insights": ["insight1", "insight2"],
        "estimated_duration": "short|medium|long"
    }}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 for better analysis
            messages=[
                {"role": "system", "content": "You are an expert content analyzer. Return only valid JSON with no additional text or formatting."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Parse and validate JSON
        parsed_result = json.loads(result)
        
        # Ensure all required fields exist
        required_fields = ["main_topics", "key_concepts", "content_type", "difficulty_level", 
                          "subject_area", "important_facts", "actionable_insights"]
        
        for field in required_fields:
            if field not in parsed_result:
                parsed_result[field] = ["Not identified"] if field.endswith('s') else "general"
        
        return parsed_result
        
    except Exception as e:
        print(f"Content analysis failed: {e}")
        # Fallback basic analysis
        return {
            "main_topics": ["General content"],
            "key_concepts": ["Key information"],
            "content_type": "educational",
            "difficulty_level": "intermediate",
            "subject_area": "general",
            "important_facts": ["Important information from video"],
            "actionable_insights": ["Key takeaways"],
            "estimated_duration": "medium"
        }

def process_multiple_chunks(chunks: List[str], count: int) -> str:
    """Process multiple chunks of text to get comprehensive coverage"""
    
    if len(chunks) == 1:
        return chunks[0]
    
    # Take first chunk + summary of others for longer content
    main_chunk = chunks[0]
    
    if len(chunks) > 1:
        # Determine how many additional chunks to process based on question count
        additional_chunks = min(2, len(chunks) - 1)
        if count > 20:
            additional_chunks = min(3, len(chunks) - 1)
        
        # Summarize remaining chunks
        remaining_text = " ".join(chunks[1:additional_chunks + 1])
        
        if len(remaining_text) > 100:  # Only summarize if there's substantial content
            summary_prompt = f"""Summarize the key points from this transcript segment in 2-3 sentences:

{remaining_text[:2000]}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize key points concisely and clearly."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                return f"{main_chunk}\n\nAdditional key points from the video: {summary}"
                
            except Exception as e:
                print(f"Chunk summarization failed: {e}")
                return main_chunk
    
    return main_chunk

def generate_advanced_qa(content_analysis: Dict, full_text: str, count: int) -> List[Dict]:
    """Generate advanced Q&A pairs using content analysis and Bloom's taxonomy"""
    
    # Calculate distribution of question types
    total_questions = count
    conceptual_questions = max(1, total_questions // 3)
    application_questions = max(1, total_questions // 3)
    analysis_questions = max(1, total_questions // 4)
    remaining = total_questions - (conceptual_questions + application_questions + analysis_questions)
    
    # Enhanced prompt with Bloom's taxonomy and content analysis
    enhanced_prompt = f"""Based on the following video transcript and content analysis, generate exactly {count} high-quality educational question-answer pairs.

CONTENT ANALYSIS:
- Main Topics: {', '.join(content_analysis.get('main_topics', []))}
- Key Concepts: {', '.join(content_analysis.get('key_concepts', []))}
- Content Type: {content_analysis.get('content_type', 'educational')}
- Difficulty Level: {content_analysis.get('difficulty_level', 'intermediate')}
- Subject Area: {content_analysis.get('subject_area', 'general')}

QUESTION DISTRIBUTION (follow this exactly):
- {conceptual_questions} Conceptual Questions (Remember/Understand level)
- {application_questions} Application Questions (Apply/Analyze level)
- {analysis_questions} Analysis Questions (Evaluate/Create level)
- {remaining} Mixed Questions

QUESTION TYPES TO INCLUDE:
1. **Conceptual**: "What is...", "How does...", "Why is...", "Explain the concept of..."
2. **Application**: "How would you apply...", "What would happen if...", "In what scenario would you use..."
3. **Analysis**: "Compare and contrast...", "Evaluate the effectiveness of...", "What are the implications of..."
4. **Scenario-based**: Real-world application questions
5. **Critical thinking**: Questions that require reasoning and evaluation

ANSWER REQUIREMENTS:
- Comprehensive but concise (3-5 sentences)
- Include specific examples from the video when possible
- Explain the "why" behind concepts
- Connect to real-world applications
- Use clear, educational language
- Avoid generic answers - be specific to the content

STRICT FORMAT (return ONLY this JSON):
[
  {{
    "question": "Question text here",
    "answer": "Comprehensive answer with explanation and examples",
    "type": "conceptual|application|analysis|scenario|critical_thinking",
    "difficulty": "easy|medium|hard",
    "topic": "main topic covered"
  }}
]

VIDEO TRANSCRIPT:
{full_text[:4000]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 for better quality
            messages=[
                {"role": "system", "content": "You are an expert educational content creator specializing in creating high-quality questions that promote deep learning and critical thinking. Always return valid JSON arrays only, with no additional text or formatting."},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )

        result = response.choices[0].message.content.strip()
        result = result.replace('```json', '').replace('```', '').strip()
        
        qa_pairs = json.loads(result)
        
        # Validate the result
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            raise ValueError("Invalid Q&A format returned")
        
        # Ensure each Q&A pair has required fields
        for i, qa in enumerate(qa_pairs):
            if not all(key in qa for key in ['question', 'answer']):
                qa_pairs[i] = {
                    'question': qa.get('question', f'Question {i+1} about the video content'),
                    'answer': qa.get('answer', 'Answer based on video content'),
                    'type': qa.get('type', 'general'),
                    'difficulty': qa.get('difficulty', 'medium'),
                    'topic': qa.get('topic', 'general')
                }
        
        return qa_pairs
        
    except Exception as e:
        print(f"Advanced Q&A generation failed: {e}")
        # Fallback to simpler generation
        return generate_fallback_qa(full_text, count)

def generate_fallback_qa(full_text: str, count: int) -> List[Dict]:
    """Fallback Q&A generation with basic improvements"""
    
    # Truncate text if too long
    text_to_use = full_text[:3500] if len(full_text) > 3500 else full_text
    
    fallback_prompt = f"""Generate exactly {count} educational question-answer pairs from this transcript.

REQUIREMENTS:
- Mix of question types: factual, conceptual, and application-based
- Answers should be 3-4 sentences with explanations
- Include specific examples from the content when possible
- Avoid yes/no questions
- Make questions educational and test understanding
- Vary difficulty levels

Return ONLY valid JSON:
[
  {{"question": "Question text", "answer": "Detailed answer with explanation", "type": "factual|conceptual|application", "difficulty": "easy|medium|hard"}}
]

Transcript: {text_to_use}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator. Return only valid JSON arrays with no additional text."},
                {"role": "user", "content": fallback_prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        result = result.replace('```json', '').replace('```', '').strip()
        
        qa_pairs = json.loads(result)
        
        # Validate and fix format
        if isinstance(qa_pairs, list):
            for i, qa in enumerate(qa_pairs):
                if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
                    qa_pairs[i] = {
                        'question': f'What are the main points discussed in section {i+1}?',
                        'answer': 'The video discusses several important concepts as outlined in the transcript.',
                        'type': 'general',
                        'difficulty': 'medium'
                    }
        else:
            raise ValueError("Invalid format returned")
            
        return qa_pairs
        
    except Exception as e:
        print(f"Fallback Q&A generation failed: {e}")
        # Ultimate fallback
        return [{
            "question": "What are the main points discussed in this video?", 
            "answer": "The video covers several important topics as outlined in the transcript. The content provides valuable information and insights on the subject matter.", 
            "type": "general", 
            "difficulty": "easy"
        }]

@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        url = data.get('url', '').strip()
        count = int(data.get('count', 10))
        quality_mode = data.get('quality_mode', 'standard')  # 'basic', 'standard', 'advanced'

        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400

        if count < 1 or count > 50:
            return jsonify({'error': 'Question count must be between 1 and 50'}), 400

        video_id = get_video_id(url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        print(f"Processing video ID: {video_id} with {quality_mode} quality mode")

        # Get transcript with WebShare proxy
        try:
            transcript = get_transcript_with_webshare_proxy(video_id)
            print(f"Successfully fetched transcript with {len(transcript)} entries")
            
        except Exception as e:
            print(f"Transcript fetching failed: {str(e)}")
            return jsonify({
                'error': f"Could not fetch transcript: {str(e)}",
                'suggestion': 'Video might not have captions, or all proxy methods failed. Try a different video.',
                'video_id': video_id,
                'proxy_status': 'WebShare proxies attempted'
            }), 500

        # Process transcript
        full_text = " ".join([entry.get('text', '') for entry in transcript])

        if len(full_text.strip()) < 100:
            return jsonify({'error': 'Transcript too short to generate meaningful questions'}), 400

        chunks = chunk_text(full_text, max_chunk_size=4000)  # Increased chunk size
        
        # Process content based on quality mode
        if quality_mode == 'advanced':
            print("Using advanced Q&A generation...")
            
            # 1. Analyze content structure
            content_analysis = analyze_content_structure(full_text)
            print(f"Content analysis: {content_analysis.get('main_topics', [])} - {content_analysis.get('content_type', 'unknown')}")
            
            # 2. Process multiple chunks for comprehensive coverage
            processed_text = process_multiple_chunks(chunks, count)
            
            # 3. Generate advanced Q&A
            qa_pairs = generate_advanced_qa(content_analysis, processed_text, count)
            
            return jsonify({
                'result': json.dumps(qa_pairs, indent=2),
                'count': len(qa_pairs),
                'video_id': video_id,
                'transcript_length': len(full_text),
                'quality_mode': 'advanced',
                'content_analysis': content_analysis,
                'chunks_processed': min(3, len(chunks)),
                'proxy_method': 'WebShare proxies'
            })
            
        elif quality_mode == 'standard':
            print("Using standard Q&A generation...")
            
            # Use improved prompt but simpler processing
            processed_text = process_multiple_chunks(chunks[:2], count)  # Use 2 chunks
            
            # Enhanced standard prompt
            enhanced_prompt = f"""Generate exactly {count} high-quality educational question-answer pairs from this video transcript.

REQUIREMENTS:
- Create a mix of question types:
  * Factual questions (What, When, Where)
  * Conceptual questions (How, Why)
  * Application questions (real-world scenarios)
- Each answer should be 3-4 sentences with clear explanations
- Include specific examples from the content when possible
- Vary difficulty levels (easy, medium, hard)
- Focus on the most important concepts
- Avoid yes/no questions

STRICT FORMAT (return ONLY valid JSON):
[
  {{
    "question": "Clear, specific question about the content",
    "answer": "Comprehensive answer with explanation and examples from the video",
    "type": "factual|conceptual|application",
    "difficulty": "easy|medium|hard"
  }}
]

Video Transcript:
{processed_text[:4000]}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",  # Better model for standard mode
                    messages=[
                        {"role": "system", "content": "You are an expert educational content creator. Generate diverse, high-quality questions that test different levels of understanding. Always return valid JSON arrays only, with no additional text or formatting."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    max_tokens=3500,
                    temperature=0.7
                )

                result = response.choices[0].message.content.strip()
                result = result.replace('```json', '').replace('```', '').strip()
                qa_pairs = json.loads(result)
                
                return jsonify({
                    'result': json.dumps(qa_pairs, indent=2),
                    'count': len(qa_pairs),
                    'video_id': video_id,
                    'transcript_length': len(full_text),
                    'quality_mode': 'standard',
                    'chunks_processed': min(2, len(chunks)),
                    'proxy_method': 'WebShare proxies'
                })

            except json.JSONDecodeError as e:
                print(f"JSON decode error in standard mode: {e}")
                # Fallback to basic generation
                qa_pairs = generate_fallback_qa(processed_text, count)
                return jsonify({
                    'result': json.dumps(qa_pairs, indent=2),
                    'count': len(qa_pairs),
                    'video_id': video_id,
                    'transcript_length': len(full_text),
                    'quality_mode': 'standard_fallback',
                    'warning': 'Used fallback generation due to JSON parsing error'
                })
            except Exception as e:
                return jsonify({'error': f'Standard mode error: {str(e)}'}), 500
        
        else:  # basic mode (original functionality with slight improvements)
            print("Using basic Q&A generation...")
            text_to_process = chunks[0]

            # Slightly improved basic prompt
            basic_prompt = f"""Generate exactly {count} educational question-answer pairs from this video transcript.

Make questions clear and educational. Each answer should be 2-3 sentences with good explanations.
Focus on the main concepts and important information from the video.

Return ONLY valid JSON:
[
  {{"question": "Question text", "answer": "Clear answer with explanation"}}
]

Video Transcript:
{text_to_process}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an educational content generator. Always return valid JSON arrays only, with no additional text or formatting."},
                        {"role": "user", "content": basic_prompt}
                    ],
                    max_tokens=2500,
                    temperature=0.7
                )

                result = response.choices[0].message.content.strip()
                result = result.replace('```json', '').replace('```', '').strip()
                
                # Validate JSON
                qa_pairs = json.loads(result)
                
                return jsonify({
                    'result': result,
                    'count': len(qa_pairs),
                    'video_id': video_id,
                    'transcript_length': len(full_text),
                    'quality_mode': 'basic',
                    'proxy_method': 'WebShare proxies'
                })

            except json.JSONDecodeError as e:
                return jsonify({'error': 'AI returned invalid JSON format', 'details': str(e)}), 500
            except Exception as e:
                return jsonify({'error': f'Basic mode error: {str(e)}'}), 500

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Server is running',
        'openai_configured': bool(openai_api_key),
        'proxy_configured': True,
        'proxy_count': len(WEBSHARE_PROXIES),
        'proxy_provider': 'WebShare',
        'features': {
            'basic_qa': True,
            'standard_qa': True,
            'advanced_qa': True,
            'content_analysis': True,
            'multi_chunk_processing': True
        }
    })

@app.route('/test_proxy', methods=['GET'])
def test_proxy():
    """Test WebShare proxy connection"""
    try:
        proxy_dict = get_random_proxy()
        
        # Test the proxy with a simple request
        response = requests.get(
            'https://httpbin.org/ip', 
            proxies=proxy_dict, 
            timeout=10
        )
        
        if response.status_code == 200:
            ip_info = response.json()
            return jsonify({
                'success': True,
                'proxy_ip': ip_info.get('origin'),
                'message': 'WebShare proxy is working!',
                'proxy_used': proxy_dict['http'].split('@')[1]  # Show IP without credentials
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Proxy returned status code: {response.status_code}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Proxy test failed: {str(e)}'
        }), 500

@app.route('/test_transcript/<video_id>', methods=['GET'])
def test_transcript(video_id):
    """Test transcript fetching with WebShare proxy"""
    try:
        transcript = get_transcript_with_webshare_proxy(video_id)
        full_text = " ".join([entry.get('text', '') for entry in transcript])
        return jsonify({
            'success': True,
            'video_id': video_id,
            'transcript_length': len(full_text),
            'transcript_entries': len(transcript),
            'preview': full_text[:300] + "..." if len(full_text) > 300 else full_text,
            'method_used': 'WebShare proxies'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'video_id': video_id,
            'error': str(e)
        }), 500

@app.route('/test_qa_quality/<video_id>/<mode>', methods=['GET'])
def test_qa_quality(video_id, mode):
    """Test different Q&A generation modes"""
    try:
        # Get a small sample for testing
        transcript = get_transcript_with_webshare_proxy(video_id)
        full_text = " ".join([entry.get('text', '') for entry in transcript])
        
        if mode == 'advanced':
            content_analysis = analyze_content_structure(full_text)
            chunks = chunk_text(full_text, max_chunk_size=4000)
            processed_text = process_multiple_chunks(chunks[:2], 5)
            qa_pairs = generate_advanced_qa(content_analysis, processed_text, 5)
            
            return jsonify({
                'success': True,
                'mode': mode,
                'video_id': video_id,
                'content_analysis': content_analysis,
                'sample_questions': qa_pairs[:3],
                'total_generated': len(qa_pairs),
                'transcript_length': len(full_text)
            })
            
        elif mode == 'standard':
            chunks = chunk_text(full_text, max_chunk_size=4000)
            processed_text = process_multiple_chunks(chunks[:2], 5)
            qa_pairs = generate_fallback_qa(processed_text, 5)
            
            return jsonify({
                'success': True,
                'mode': mode,
                'video_id': video_id,
                'sample_questions': qa_pairs[:3],
                'total_generated': len(qa_pairs),
                'transcript_length': len(full_text)
            })
            
        elif mode == 'basic':
            chunks = chunk_text(full_text, max_chunk_size=3000)
            qa_pairs = generate_fallback_qa(chunks[0], 3)
            
            return jsonify({
                'success': True,
                'mode': mode,
                'video_id': video_id,
                'sample_questions': qa_pairs,
                'total_generated': len(qa_pairs),
                'transcript_length': len(full_text)
            })
            
        else:
            return jsonify({'error': 'Invalid mode. Use: basic, standard, or advanced'}), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'mode': mode,
            'video_id': video_id
        }), 500

@app.route('/analyze_content/<video_id>', methods=['GET'])
def analyze_content(video_id):
    """Analyze video content without generating Q&A"""
    try:
        transcript = get_transcript_with_webshare_proxy(video_id)
        full_text = " ".join([entry.get('text', '') for entry in transcript])
        
        content_analysis = analyze_content_structure(full_text)
        chunks = chunk_text(full_text, max_chunk_size=4000)
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'transcript_length': len(full_text),
            'transcript_entries': len(transcript),
            'content_analysis': content_analysis,
            'chunks_available': len(chunks),
            'estimated_questions': {
                'basic': '5-10 questions',
                'standard': '10-20 questions',
                'advanced': '15-30 questions'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'video_id': video_id,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY not set!")
    else:
        print("✅ OpenAI API key configured")
    
    print(f"✅ WebShare proxies configured: {len(WEBSHARE_PROXIES)} proxies available")
    print("🚀 Starting improved YouTube Q&A Generator with multiple quality modes...")
    print("📋 Available modes: basic, standard, advanced")
    print("🔧 Available endpoints:")
    print("  - POST /generate_qa (main endpoint)")
    print("  - GET /health (health check)")
    print("  - GET /test_proxy (test proxy connection)")
    print("  - GET /test_transcript/<video_id> (test transcript fetching)")
    print("  - GET /test_qa_quality/<video_id>/<mode> (test Q&A generation)")
    print("  - GET /analyze_content/<video_id> (analyze content without generating Q&A)")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)