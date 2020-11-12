from    __init__   import   interactive, question, setup # tried . did'nt work
from    argparse   import   ArgumentParser
parser = ArgumentParser(description = 'Choose what you want to do')
parser.add_argument('-s', '--Setup', action='store_true', help='Setup the requirements for the code')
group = parser.add_mutually_exclusive_group()
group.add_argument('-i', '--Interactive', action='store_true', help='Open an interactive session for several questions')
group.add_argument('-q', '--Question', metavar='', type=str, help='Ask a single question from the command line')
args = parser.parse_args()
if args.Setup: setup()
if args.Interactive: interactive()
elif args.Question: question(args.Question)
elif not args.Setup: print('You didn\'t give me any input, to ask a single question type \"-q Question\" not \"-q\"')
