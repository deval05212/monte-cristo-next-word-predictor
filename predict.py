import time
from model import NextWordPredictor

def main():
    predictor = NextWordPredictor()
    
    try:
        predictor.load_existing_model()
    except Exception as e:
        print("Error loading model. Make sure you have trained it first.")
        print(f"Details: {e}")
        return

    print("Prediction Phase Ready")
    
    # Interactive prediction mode
    print("Interactive Prediction Mode (type 'exit' to quit)")
    while True:
        user_input = input("Enter a phrase: ")
        if user_input.strip().lower() == 'exit':
            print("Exiting...")
            break
            
        if user_input.strip():
            print(f"Prediction: {user_input}", end="", flush=True)
            
            current_text = user_input
            for _ in range(50):
                next_word = predictor.predict_next_word(current_text)
                if not next_word:
                    break
                
                print(f" {next_word}", end="", flush=True)
                current_text += " " + next_word
                time.sleep(0.01)
            print('\n')

if __name__ == "__main__":
    main()
