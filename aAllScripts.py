import os
import pygame
import importlib
import time

def main():
	while True:
		print("\nHello! This is a compilation of all of the code made throughout the trimester, sorted by person.")
		print("In order to select a person to see their code, type in the number corrisponding to them, and hit RETURN.\n")
		p = input("Input the name of the student you would like to see, with the first letter capitalized (ex. Rey not rey) >>> ")
		os.system('clear')
		try:
		 	func = importlib.import_module(p)
		 	func.main()
		except:
			print("Something went wrong, please try again. ")
		input("\n\n\npress RETURN to go back to the beginning")
		os.system('clear')

main()
