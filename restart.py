import os
import sys

def restart_program():
    """Restarts the current program."""
    print("Restarting the program...")
    os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    # Your main code here
    print("Running the program...")
    
    # Restart program at the end of execution
    restart_program()
