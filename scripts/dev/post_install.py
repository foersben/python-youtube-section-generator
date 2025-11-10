"""Poetry post-install hook for interactive setup."""
import scripts.setup_interactive as setup_interactive

def main():
    setup_interactive.main()

if __name__ == "__main__":
    main()

