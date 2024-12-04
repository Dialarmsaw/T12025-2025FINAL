def main():
    choice = input("Press and enter 1 for a dollars to euros conversion, press and enter 2 for code about horror films, press and enter 3 for code about probabilities in books, press and enter 4 for code about musical artists")
    if choice == "1":
        M = 1.08
        euro = int(input("How many USD do you have?"))
        print("You have", M * euro, "euros")
    if choice == "2":
        Subgenre1 = input ("What kind of horror film do you want to watch?")
        if Subgenre1 == "Slasher":
            print ("You might like Halloween (1978), Friday the 13th (1980), or Black Christmas (1974).")
        if Subgenre1 == "Found Footage":
            print ("You might like The Blair Witch Project (1999), Paranormal Activity (2007), or V/H/S (2012).")
        if Subgenre1 == "Creature Feature":
            print ("You might like Dracula (1931), Frankenstein (1931), or The Creature From The Black Lagoon (1954).")
        if Subgenre1 == "Posession":
            print ("You might like The Exorcist (1973), The Conjuring (2013), or Jennifer's Body (2009).")
        if Subgenre1 == "Aliens":
            print ("You might like Alien (1979), Signs (2002), or The Thing (1982).")
        if Subgenre1 == "Comedy":
            print ("You might like Lisa Frankenstein (2024), Ghostbusters (1984), or Zombieland (2009).")
        if Subgenre1 == "Body Horror":
            print ("You might like Hellraiser (1987), The Fly (1988), or Teeth (2007).")
        if Subgenre1 == "Classic":
            print ("You might like Psycho (1960), The Shining (1980), or Rosemary's Baby (1968).")
        if Subgenre1 == "Teen":
            print ("You might like The Craft (1996), Urban Legend (1998), or Final Destination (2000).")
        if Subgenre1 == "Vampire":
            print ("You might like Fright Night (1985), The Lost Boys (1987), or 30 Days of Night (2007).")
        if Subgenre1 == "Kid-Friendly":
            print ("You might like Coraline (2009), The Addams Family (1991), or Frankenweenie (2012).")
        if Subgenre1 == "Zombie":
            print ("You might like Night of the Living Dead (1968), The Return of the Living Dead (1985), or Train to Busan (2016).")
        if Subgenre1 == "Cult Classic":
            print ("You might like Serial Mom (1994), Killer Klowns from Outer Space (1988), or Re-Animator (1985).")
    if choice == "3":
        import random
        b = int (input ("How many pages is your book?"))
        success = 0
        p = int (input ("What number do you want your page number to be divisible by?"))
        for _ in range(10000):
            n = random.randrange (1,b+1)
            if n%p == 0:
                success+=1
        print(success/10000)
    if choice == "4":
        Genre1 = input ("What genre of music do you want to listen to today?")
        print ("You want to listen to", Genre1)
        if Genre1 == "Pop":
            print ("You should listen to Chappell Roan, Sabrina Carpenter, or Charli XCX.")
        if Genre1 == "Rock":
            print ("You should listen to Queen, David Bowie, or the Rolling Stones.")
        if Genre1 == "Alternative":
            print ("You should listen to Hozier, Fleetwood Mac, or Billie Eilish.")
        if Genre1 == "Country":
            print ("You should listen to Marren Morris, Dolly Parton, or Carrie Underwood.")
        if Genre1 == "Hip Hop" "Rap":
            print ("You should listen to Eminem, Kendrick Lamar, or Lil Wayne.")
        if Genre1 == "Metal":
            print ("You should listen to Ice Nine Kills, Black Sabbath, or Korn.")
        if Genre1 == "K-pop":
            print ("You should listen to BTS, Stray Kids, or BlackPink.")
        if Genre1 == "Pop":
            print ("Check out the albums 'The Rise and Fall of a Midwest Princess', 'Short n' Sweet', or 'Brat'.")
        if Genre1 == "Rock":
            print ("Check out the albums 'Queen's Greatest Hits', 'Ziggy Stardust', or 'Out of Our Heads'.")
        if Genre1 == "Alternative":
            print ("Check out the albums 'Wasteland, Baby!', 'Rumours', or 'When We All Fall Asleep Where Do We Go?'.")
        if Genre1 == "Country":
            print ("Check out the albums 'Girl', 'Joelene', or 'Some Hearts'.")
        if Genre1 == "Hip Hop" "Rap":
            print ("Check out the albums 'The Eminem Show', 'To Pimp A Butterfly', or 'Funeral'.")
        if Genre1 == "Metal":
            print ("Check out the albums 'The Silver Scream', 'Paranoid', or 'Korn'.")
        if Genre1 == "K-pop":
            print ("Check out the albums 'Map of the Soul: 7', 'Go Live', or 'The Album'.")

