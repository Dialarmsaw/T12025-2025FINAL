def main():
    from datetime import datetime

    # Get the user's birthdate
    birth_date = input("Enter your birthdate (YYYY-MM-DD): ")

    # Calculate age in seconds
    age_seconds = (datetime.now() - datetime.strptime(birth_date, "%Y-%m-%d")).total_seconds()

    # Display the result
    print("You are approximately ", str(age_seconds), "seconds old.")


