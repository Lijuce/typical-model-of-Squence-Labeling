import sys
import argparse
from argparse import RawTextHelpFormatter
import ast

def parse_arguments(arguments=None):
    ''' Parse the arguments

    arguments:
        arguments the arguments, optionally given as argument
    '''
    argparser = argparse.ArgumentParser(description='''seq2seq for NER''',
                                        formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--input_file', required=True)
    argparser.add_argument('--output_file', required=False)
    argparser.add_argument('--train', required=True, type=ast.literal_eval, choices=[True, False])
    argparser.add_argument('--model_file', required=True)
    argparser.add_argument('--epochs', type=int, default=5, required=False)
    argparser.add_argument('--lr', type=float, default=0.001, required=False)
    argparser.add_argument('--crf', type=ast.literal_eval, default=False, required=False, choices=[True, False])
    argparser.add_argument('--log_file', default='./log', required=False)
    
    try:
        arguments = argparser.parse_args(args=arguments)
        
    except:
        argparser.print_help()
        sys.exit(0)

    arguments = vars(arguments) 
    return arguments
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = parse_arguments(sys.argv[1:])
    else:
        arguments = parse_arguments()
    print(arguments)
    if arguments['train']:
        print("training")
    else:
        print("evaluating")


