def main():
    S = int(input("Hi!! Would you like to play Supermarket (1) or Hangman (2)"))

    if S == 1:
        import random

        # Constants (times are in seconds)
        ITEM_SCAN_TIME = 2            # Time to scan each item
        TRANSACTION_COMPLETE_TIME = 30  # Time to complete payment and bagging

        class Shopper:
            def __init__(self, items, is_express=False):
                self.items = items
                self.is_express = is_express
                self.total_time = items * ITEM_SCAN_TIME + TRANSACTION_COMPLETE_TIME

        class Cashier:
            def __init__(self, is_express=False, on_break=False):
                self.is_express = is_express
                self.on_break = on_break
                self.line = []

            def add_shopper(self, shopper):
                self.line.append(shopper)

            def line_wait_time(self):
                if self.on_break:
                    return float('inf')  # Unavailable cashier line has infinite wait time
                return sum(shopper.total_time for shopper in self.line)

        def assign_shopper_to_best_line(shoppers, cashiers):
            for shopper in shoppers:
                # Select cashiers based on shopper type (express or regular)
                available_cashiers = [cashier for cashier in cashiers if not cashier.on_break and (cashier.is_express == shopper.is_express)]
                
                # Find the line with the minimum wait time
                best_cashier = min(available_cashiers, key=lambda c: c.line_wait_time(), default=None)
                
                if best_cashier:
                    best_cashier.add_shopper(shopper)
                else:
                    print(f"No available line for {'express' if shopper.is_express else 'regular'} shopper with {shopper.items} items.")

        def irl_supermarket():
            # Gather user inputs
            num_cashiers = int(input("Enter the number of cashiers: "))
            
            
            num_shoppers = int(input("Enter the number of shoppers: "))
            max_items_per_shopper = int(input("Enter the maximum number of items a shopper can have: "))
            express_line_probability = float(input("Enter the probability that a cashier is an express line (0-1)(i.e 50% = 0.5): "))
            break_probability = float(input("Enter the probability that a cashier is on break (0-1)(i.e 50% = 0.5): "))
            if break_probability == 1:
                print("")
                print("")
                print("")
                print("")
                print("Store closed no one is working ):")
                print("(check if probability = 1 cause then 100% of people are not working)")
                print("")
                print("")
                print("")
                print("Information:")
                print("")
                
                

            # Create cashiers with some express lines based on the given probability
            cashiers = [Cashier(is_express=(random.random() < express_line_probability)) for _ in range(num_cashiers)]
            
            # Randomly assign some cashiers to go on break based on the given probability
            for cashier in cashiers:
                cashier.on_break = (random.random() < break_probability)

            # Create shoppers with random item counts and express eligibility
            shoppers = [Shopper(items=random.randint(1, max_items_per_shopper), is_express=random.choice([True, False])) for _ in range(num_shoppers)]
            
            # Assign each shopper to the best line
            assign_shopper_to_best_line(shoppers, cashiers)
            
            # Display results
            for i, cashier in enumerate(cashiers):
                print(f"Cashier {i + 1} ({'Express' if cashier.is_express else 'Regular'}) {'(On Break)' if cashier.on_break else ''}:")
                if cashier.on_break:
                    continue
                print(f"  Line Length: {len(cashier.line)} shoppers")
                print(f"  Estimated Total Wait Time: {cashier.line_wait_time()} seconds\n")

        # Run the supermarket simulation
        irl_supermarket()

    if S == 2:
        import random
        hangman_stages = [
        """
        ~
        (..)

        """,
        
        """
        ~
        (..)
        |

    
        """,
        
        """
        ~
        (..)
        -
        _/| 
            
        
        """,
        
        """
        ~
        (..)
        -
        _/| \_
        |  
        
        """,
        
        """
        ~
        (..)
        -
        _/| \_
        |  
        / 
    _/   
    
        """,
        
        """
        ~
        (..)
        -
        _/| \_
        |  
        /\ 
        _/  \_ 

    
    
        """
        ]

        # List of words to choose from
        words = ["pumpkin", "spider", "scull", "skeleton", "spooky", "nightmare before christmas", "candy corn"]

        def get_random_word(words):
        #Selects a random word from the list.
            return random.choice(words)

        def display_progress(word, guessed_letters):
            #Displays the current progress of the guessed word
            display = [letter if letter in guessed_letters else "_" for letter in word]
            return " ".join(display)

        def play_hangman():
            word = get_random_word(words)  # select a random word
            guessed_letters = set()   # to keep track of guessed letters
            attempts_left = len(hangman_stages) -1  # Number of attempts allowed

            print( )
            print("Happy Halloween!! Welcome to Hangman! Try an guess a spooky word.")
            
            while attempts_left != 0:
                print("\nCurrent word:", display_progress(word, guessed_letters))
                print(hangman_stages[attempts_left])  # Display the current hangman stage
                guess = input("Guess a letter: ").lower()

                if guess in guessed_letters:
                    print("You already guessed that letter.")
                elif guess in word:
                    guessed_letters.add(guess)
                    print("Good guess!")
                    # Check if the player has won
                    if all(letter in guessed_letters for letter in word):
                        print("\nCongratulations! You've guessed the word:", word)
                        break
                else:
                    attempts_left -= 1
                    guessed_letters.add(guess)
                    print("Wrong guess! Attempts remaining:", attempts_left)

            else:
                print("\nYou lose! The word was:", word)
                print(hangman_stages[0])  # Display the full hangman

            # Ask if the player wants to play again
            play_again = input("Would you like to play again? (yes/no): ").lower()
            if play_again == "yes":
                play_hangman()
            else:
                print("Thanks for hanging!")

        # Start the game
        play_hangman()



            

