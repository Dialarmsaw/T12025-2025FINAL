def main():
    if input("Press and enter 1 for code about ages, press and enter 2 for code about passwords") == "1":
        #determines your age range

        age = float(input("How old are you? "))
        if age >= 61 and age <= 100:
            print("You are a senior citizen.")
        elif age >= 18 and age <=60:
            print("You are an adult.")
        elif age >= 13 and age <= 17:
            print("You are a teenager.")
        else:
            print("You are a child.")
    else:
        # Password Generator

        import random

        letters = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        symbols = ['!', '#', '$', '%', '&', '*', '+']

        print("Welcome to the Password Generator!")
        nr_letters = int(input("How many letters would you like in your password?\n"))
        nr_symbols = int(input("How many symbols would you like?\n"))
        nr_numbers = int(input("How many numbers would you like?\n"))

        password_list = []

        for char in range(0, nr_letters):
            password_list.append(random.choice(letters))

        for char in range(0, nr_symbols):
            password_list.append(random.choice(symbols))

        for char in range(0, nr_numbers):
            password_list.append(random.choice(numbers))

        random.shuffle(password_list)

        password = ""
        for char in password_list:
            password += char

        # convert list to string
        pwd = ''.join(password_list)
        print(f"Your random password to use is: {pwd}")