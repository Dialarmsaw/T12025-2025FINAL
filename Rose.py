#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:37:19 2024

@author: rose
"""

def main():
    s=int(input("run code 1 or 2"))

    if s == 1: 

        import random
        random_number = random.randint(1,100)
        
        print ("Cents: " + str(random_number))
        
        Quarter = 0
        Dime = 0
        Nickel = 0
        Penny = 0
        
        if random_number > 25:
            Quarter+= 1
            
        # print (Quarter , "if statement")
            #print (random_number)
        
        if random_number > 25: 
            Quarter+=1
            random_number -=25
            
            #print (Quarter ,"step two") 
        # print (random_number)
        
        if random_number > 10:
            Dime += 1
            random_number -=10
            
        if random_number > 10:
            Dime += 1
            random_number -=10
            
        if random_number > 5:
            Nickel += 1
            random_number -=5
            
        if random_number >= 5:
            Nickel += 1
            random_number -=5
            
        if random_number >= 1:
            Penny += 1
            random_number -=1
        if random_number >= 1:
            Penny += 1
        if random_number > 1:
            Penny += 1
            random_number -=1
        if random_number > 1:
            Penny += 1
        
        
        print ("Number of quarters: " + str(Quarter))
        print ("Number of dimes: " + str(Dime))
        print ("Number of nickels: " + str(Nickel))
        print ("Number of pennies: " + str(Penny))


    if s == 2: 
        
        import random 
        
        rate_queen=0 
        
        rate_three=0
        
        def deck():
            ##   Creates a deck of cards.
            suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
            numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
            card = random.choice(suits) + random.choice (numbers) ## choses random card
            return card
        
        # %%
        n=10000
        
        counter = 0
        for card in range(n):#how many times are we counting?
            cardA = deck()
            if "Q" in cardA :
                counter += 1
            cardB = deck()
        
        print ("Frequency of queens in a deck of cards:" + str(counter/n))
        
                
        for card in range(n):#how many times are we counting?
            cardB = deck()
            if "3" in cardB :
                counter += 1
            cardB = deck()
        
        print ("Frequency of 3s in a deck of cards" + str(counter/n))
