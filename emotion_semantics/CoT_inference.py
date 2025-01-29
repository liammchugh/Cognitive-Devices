import speech_recognition as sr
import time
import threading

# from emotive_classifier import ser_inference
# Example: ser_result = ser_inference.process(text_context, image_context)

# Placeholder for Gemma 2B or any small LM you might use
# from gemma2b import Gemma2B


class StrategicEmotionalAssistant:
    """
    Demonstrates a pipeline:
      1) Capture and transcribe audio
      2) Obtain emotive context via ser_inference
      3) Run chain-of-thought queries on Gemma2B
      4) Provide a final strategic suggestion
    """
    def __init__(self):
        # Placeholder for Gemma2B loading
        # self.gemma_model = Gemma2B(model_path="path/to/gemma2b")
        
        # Initialize a conversation buffer
        self.conversation_history = []
        
        # Potentially store imagery or other context
        self.image_context_buffer = []
        
        # For concurrency or background listening
        self.stop_listening = None
    
    def capture_audio_once(self):
        """
        Captures audio from the microphone once, then transcribes using `speech_recognition`.
        """
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening... (Speak now)")
            audio_data = recognizer.listen(source)
        
        # Try to transcribe
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("[Error] Speech was unintelligible")
            return None
        except sr.RequestError as e:
            print(f"[Error] Could not request results from service; {e}")
            return None
    
    def background_listen(self, callback, phrase_time_limit=5):
        """
        Starts background listening in a separate thread. 
        Calls `callback` with transcribed text each time new audio is recognized.
        """
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        def _listen_bg():
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
            
            def _callback(recognizer_instance, audio_data):
                try:
                    text = recognizer_instance.recognize_google(audio_data)
                    callback(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"[Error] Request failed; {e}")
            
            self.stop_listening = recognizer.listen_in_background(
                mic, _callback, phrase_time_limit=phrase_time_limit
            )
            # Keep the background thread alive until stopped
            while True:
                time.sleep(0.1)
        
        thread = threading.Thread(target=_listen_bg, daemon=True)
        thread.start()
    
    def stop_background_listen(self):
        """Stops the background listening thread if running."""
        if self.stop_listening is not None:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None
    
    def emotive_analysis(self, text_context, image_context=None):
        """
        Example function to run your SER (speech emotion recognition) or emotional classifier
        and return results. Replace with your actual `ser_inference.process()`.
        """
        if image_context is None:
            image_context = []
        
        # Placeholder call to your own SER model
        # ser_result = ser_inference.process(text_context, image_context)
        
        # Return a mocked result for now
        ser_result = {
            'emotion': 'neutral',
            'confidence': 0.85
        }
        return ser_result
    
    def gemma_chain_of_thought(self, text_context, ser_result):
        """
        Example multi-query chain-of-thought approach with Gemma2B.
        This might involve repeated queries to the model with reasoning prompts.
        """
        # Combine the text context and emotion label for step-by-step prompting
        # Placeholder chain-of-thought style queries:
        
        # Example of a thought chain
        # 1) Summarize context
        # 2) Evaluate emotional content
        # 3) Provide strategic suggestion
        # 
        # Replace with your actual calls to gemma2b:
        
        # Example pseudo code:
        # step1_prompt = f"Summarize the following conversation:\n\n{text_context}"
        # step1_result = self.gemma_model.generate(step1_prompt)
        # 
        # step2_prompt = f"User emotion: {ser_result['emotion']} with confidence {ser_result['confidence']}. "\
        #               f"Reflect on emotional implications for the user:\n\n{step1_result}"
        # step2_result = self.gemma_model.generate(step2_prompt)
        # 
        # step3_prompt = f"Given the user context and emotional state, provide a strategic response:\n\n{step2_result}"
        # step3_result = self.gemma_model.generate(step3_prompt)
        # 
        # For now, let's mock the final output:
        
        chain_of_thought = [
            "Step1: Summarize the conversation => (mock summary)",
            f"Step2: The user seems {ser_result['emotion']} => (mock emotional analysis)",
            "Step3: Provide a strategic suggestion => (mock final suggestion)"
        ]
        
        # The final suggestion is presumably the last step
        final_suggestion = "Remember to empathize and validate the user’s feelings before proposing solutions."
        
        return chain_of_thought, final_suggestion
    
    def add_to_conversation(self, text):
        """Add new text to the conversation history."""
        self.conversation_history.append(text)
    
    def handle_new_input(self, text):
        """
        When new text input is received (from transcription or typed),
        1) Add to conversation context
        2) Analyze emotion
        3) Get chain-of-thought suggestions from Gemma2B
        4) Print or return the final strategic suggestion
        """
        if not text:
            return
        
        # 1) Add new text to conversation
        self.add_to_conversation(text)
        
        # 2) Emotive analysis
        full_context = "\n".join(self.conversation_history)
        ser_result = self.emotive_analysis(full_context, self.image_context_buffer)
        
        # 3) Chain-of-thought with Gemma
        chain_of_thought, suggestion = self.gemma_chain_of_thought(full_context, ser_result)
        
        # 4) Print or return the strategic suggestion
        print("=== Chain of Thought ===")
        for step in chain_of_thought:
            print(step)
        print("=== Final Suggestion ===")
        print(suggestion)
        print("========================\n")
    
    def run_once(self):
        """
        Captures one audio snippet, transcribes, processes, then prints a suggestion.
        """
        print("Starting single round capture...")
        transcribed_text = self.capture_audio_once()
        
        if transcribed_text:
            print(f"[User said]: {transcribed_text}")
            self.handle_new_input(transcribed_text)
        else:
            print("[No valid transcription]")
    
    def run_loop(self):
        """
        Continuously capture audio in a loop (blocking version).
        Press Ctrl+C to stop.
        """
        try:
            while True:
                self.run_once()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting run_loop...")

        
if __name__ == "__main__":
    # Instantiate the assistant
    assistant = StrategicEmotionalAssistant()
    
    # Option 1: Do a single run
    assistant.run_once()
    
    # Option 2: Listen in a loop
    # assistant.run_loop()
    
    # Option 3: Background listening
    # def callback(text):
    #     print(f"[Heard in background]: {text}")
    #     assistant.handle_new_input(text)
    #
    # assistant.background_listen(callback, phrase_time_limit=5)
    # while True:
    #     try:
    #         time.sleep(1)
    #     except KeyboardInterrupt:
    #         print("\nStopping background listening...")
    #         assistant.stop_background_listen()
    #         break
