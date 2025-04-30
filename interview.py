import streamlit as st
import av
import asyncio
import json
import time
import numpy as np
import requests
import io
import wave
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import google.generativeai as genai
from google.api_core import retry_async, exceptions

# Configure APIs
# IMPORTANT: Replace with your actual API keys
genai.configure(api_key="")
ASSEMBLY_API_KEY = ""

# Custom CSS remains the same as in the original script
st.markdown("""
<style>
    /* All previous CSS styling from the original script */
    .question-container {
        background-color: #;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    /* ... rest of the original CSS ... */
</style>
""", unsafe_allow_html=True)

class AudioProcessor(VideoProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.is_recording = False
        self.frames_received = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            if frame.format.name == 's16':
                audio_data = frame.to_ndarray()

                if self.is_recording:
                    self.frames_received += 1
                    self.audio_buffer.append(audio_data.astype(np.int16).tobytes())

            return av.VideoFrame.from_ndarray(
                np.zeros((1, 1, 3), dtype=np.uint8),
                format="rgb24"
            )
        except Exception as e:
            st.error(f"Audio processing error: {e}")
            return frame

    async def process_audio(self):
        if not self.audio_buffer or self.frames_received == 0:
            st.warning("No audio data captured. Check microphone settings.")
            return ""

        try:
            audio_chunk = b''.join(self.audio_buffer)
            wav_buffer = io.BytesIO()

            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_chunk)

            if wav_buffer.tell() < 1000:
                st.warning("Audio recording seems too short or empty.")
                return ""

            return await transcribe_audio(wav_buffer.getvalue())

        except Exception as e:
            st.error(f"Audio processing failed: {e}")
            return ""

    def start_recording(self):
        self.is_recording = True
        self.audio_buffer = []
        self.frames_received = 0

    def stop_recording(self):
        self.is_recording = False

async def transcribe_audio(audio_data):
    """Enhanced transcription with robust error handling"""
    try:
        base_url = "https://api.assemblyai.com/v2"
        headers = {
            "authorization": ASSEMBLY_API_KEY,
            "content-type": "application/json"
        }

        # Upload audio file
        upload_response = requests.post(
            f"{base_url}/upload",
            headers=headers,
            data=audio_data
        )
        upload_response.raise_for_status()
        upload_url = upload_response.json()["upload_url"]

        # Create transcription request
        transcript_request = {
            "audio_url": upload_url,
            "language_code": "en_us",
            "word_boost": [
                "professional", "experience", "project",
                "team", "skills", "background"
            ],
            "disfluencies": True,
            "punctuate": True
        }

        transcript_response = requests.post(
            f"{base_url}/transcript",
            json=transcript_request,
            headers=headers
        )
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()['id']

        # Poll for transcription
        max_attempts = 30
        for _ in range(max_attempts):
            polling_response = requests.get(
                f"{base_url}/transcript/{transcript_id}",
                headers=headers
            )
            polling_response.raise_for_status()
            transcription_result = polling_response.json()

            if transcription_result['status'] == 'completed':
                return transcription_result['text']
            elif transcription_result['status'] == 'error':
                st.error(f"Transcription failed: {transcription_result.get('error', 'Unknown error')}")
                return ""

            await asyncio.sleep(3)  # Wait between polls

        st.warning("Transcription timed out. Please try again.")
        return ""

    except requests.RequestException as e:
        st.error(f"Network error during transcription: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected transcription error: {e}")
        return ""

@retry_async.AsyncRetry()
async def generate_feedback(question, response):
    """Generate structured feedback on the response"""
    try:
        model = genai.GenerativeModel(
            "gemini-pro",
            generation_config=genai.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.6
            )
        )

        prompt = f"""Analyze this interview response to the question: "{question}"

        Response: {response}

        Provide structured feedback in these three categories:
        ✅ Strength: [positive aspect of the response]
        📈 Improvement: [area that could be improved]
        💡 Recommendation: [specific actionable advice]
        """

        result = await model.generate_content_async(prompt)
        return result.text
    except Exception as e:
        return f"Feedback Error: {str(e)}"

# The rest of the script remains exactly the same as your original implementation
# This includes main_async(), main(), and all other functions and session state management

async def main_async():
    st.title("Professional Interview Coach 🎤")

    # Initialize session state (same as original script)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(int(time.time()))
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
    if "all_answered" not in st.session_state:
        st.session_state.all_answered = False

    # Interview questions (same as original)
    questions = [
        "Tell me about yourself and your professional background",
        "Describe a challenging project you successfully managed",
        "How do you handle conflict within a team?",
        "What are your greatest strengths and how do you apply them?",
        "Where do you see yourself professionally in five years?"
    ]


    # Display progress
    progress_pct = (st.session_state.current_q / len(questions)) * 100
    st.markdown(f"""
    <div class="progress-indicator">
        Question {st.session_state.current_q + 1}/{len(questions)} - {progress_pct:.0f}% Complete
    </div>
    """, unsafe_allow_html=True)

    # Check if all questions have been answered
    if st.session_state.current_q < len(questions):
        current_question = questions[st.session_state.current_q]

        # Display the current question in a styled box
        st.markdown(f"""
        <div class="question-container">
            <h3>Question {st.session_state.current_q + 1}:</h3>
            <h4>{current_question}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Display ad blocker warning if needed
        with st.expander("⚠️ Having Audio Issues?", expanded=False):
            st.markdown("""
            - **Disable ad blockers** for this site
            - Set your volume to 50% or lower to prevent feedback
            - Make sure your microphone is enabled in browser settings
            - Use headphones to prevent audio feedback loops
            """)

        # Input mode selection
        mode = st.radio(
            "Choose your response method:",
            ["🎤 Voice Response (20 sec)", "✍️ Text Response (30 sec)"],
            horizontal=True
        )

        if "Voice" in mode:
            # Voice response section
            webrtc_ctx = webrtc_streamer(
                key=f"voice_{st.session_state.session_id}_{st.session_state.current_q}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
                },
                media_stream_constraints={
                    "video": False,
                    "audio": {
                        "sampleRate": 16000,
                        "channelCount": 1,
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": False
                    }
                },
                video_processor_factory=AudioProcessor
            )

            if webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor

                if st.button("Start Recording (20 sec)", type="primary"):
                    # Create a placeholder for the timer
                    timer_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    # Start recording
                    processor.start_recording()

                    # Run the countdown timer (20 seconds for voice)
                    for i in range(20):
                        remaining = 20 - i
                        timer_placeholder.markdown(f"""
                        <div class="timer">
                            ⏱️ {remaining} seconds remaining
                        </div>
                        """, unsafe_allow_html=True)

                        progress_bar.progress((i + 1) / 20)

                        # Warning at 2 seconds
                        if remaining == 2:
                            st.toast("⚠️ Only 2 seconds remaining!", icon="⏰")

                        await asyncio.sleep(1)

                    # Stop recording
                    processor.stop_recording()
                    timer_placeholder.empty()
                    progress_bar.empty()

                    # Process the audio
                    with st.spinner("Processing your response..."):
                        transcript = await processor.process_audio()

                    if transcript:
                        # Store response without feedback yet
                        st.session_state.history.append({
                            "question": current_question,
                            "response": transcript,
                            "feedback": None
                        })

                        # Move to next question
                        st.session_state.current_q += 1
                        if st.session_state.current_q >= len(questions):
                            st.session_state.all_answered = True
                        st.rerun()
                    else:
                        st.error("No speech detected. Please check your microphone and try again.")

        else:
            # Text response section with timer
            with st.form(f"text_response_{st.session_state.current_q}"):
                user_text = st.text_area("Type your response:", height=150)

                col1, col2 = st.columns([3, 1])
                with col1:
                    submit = st.form_submit_button("Submit Response", type="primary")

                with col2:
                    # Static timer info (actual timer runs after form submission)
                    st.markdown("""
                    <div class="timer">
                        30 second limit
                    </div>
                    """, unsafe_allow_html=True)

            # Handle submission
            if submit:
                if user_text.strip():
                    # Set a timer for animation
                    timer_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    # Simulate a brief "processing" time
                    for i in range(5):
                        progress_bar.progress((i + 1) / 5)
                        await asyncio.sleep(0.1)

                    # Store response without feedback yet
                    st.session_state.history.append({
                        "question": current_question,
                        "response": user_text,
                        "feedback": None
                    })

                    # Move to next question
                    st.session_state.current_q += 1
                    if st.session_state.current_q >= len(questions):
                        st.session_state.all_answered = True
                    st.rerun()
                else:
                    st.warning("Please enter a response before submitting.")

    # Process all responses and show feedback if all questions are answered
    elif st.session_state.all_answered:
        st.balloons()
        st.header("🏆 Complete Interview Review", divider="rainbow")

        # Process feedback for all responses if not already done
        pending_feedback = False
        for entry in st.session_state.history:
            if entry["feedback"] is None:
                pending_feedback = True

        if pending_feedback:
            with st.spinner("Analyzing all responses..."):
                # Generate feedback for all responses
                for i, entry in enumerate(st.session_state.history):
                    if entry["feedback"] is None:
                        feedback = await generate_feedback(entry["question"], entry["response"])
                        st.session_state.history[i]["feedback"] = feedback

        # Display all Q&A with feedback
        for idx, entry in enumerate(st.session_state.history):
            with st.expander(f"Question {idx + 1}: {entry['question']}", expanded=True):
                st.markdown(f"""
                <div class="response-box">
                    <strong>Your Response:</strong><br>
                    {entry['response']}
                </div>

                <div class="feedback-box">
                    <strong>AI Feedback:</strong><br>
                    {entry['feedback']}
                </div>
                """, unsafe_allow_html=True)

        # Add a reset button
        if st.button("Start New Interview", type="primary"):
            # Reset session state
            st.session_state.history = []
            st.session_state.current_q = 0
            st.session_state.all_answered = False
            st.session_state.session_id = str(int(time.time()))
            st.rerun()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
