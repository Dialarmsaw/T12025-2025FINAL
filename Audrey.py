def main():
    if input("Press and enter 1 for code about prime numbers, press and enter 2 for code about wages") == "1":
        prime=0
        rolls=0

        print("You have a die and want to see how often you get a prime number.")

        import random

        amount=int(input("how many times do you want to roll a die?"))

        for numbers in range(amount):
            rolls+=1
            numbs= random.randrange(1,7)
            print(numbs)
            if numbs == 1:
                prime+=1
            if numbs == 3:
                prime+=1
            if numbs == 5:
                prime+=1

        print("You rolled the die", rolls, "times.") 
        print("and the percent of rolls that were prime was", prime*100/rolls, "%")
        print("Experimentally the percent should be 50%")
    else: 
        Wage=int(input("What is your hourly wage?"))
        Work=int(input("How many hours did you work?")) 
        overtime=int(input("What is your regular work hours?"))   
        if Work > overtime:
            #print(Work-overtime)
            #print(Wage*1.5)
            #print((Work-overtime)*(Wage*1.5))
            #print(overtime*Wage+(Work-overtime)*(Wage*1.5))
            profit=overtime*Wage+(Work-overtime)*(Wage*1.5)
        else:
            profit=Wage*Work
        print("your salary is", profit, ".")