import logging
import territory_finder
from territory_finder import TerritoryFinder

import os, sys
from time import time
import argparse

def main():

    try:
        t0 = time()
        # Set up logging
        logger = logging.getLogger('territory_finder_application')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(u"./logs/territory_finder.log", "w")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.debug("Parsing command line")

        parser = argparse.ArgumentParser(description="Find a territory for a trade outlet")

        # Parse arguments
        parser.add_argument("-c", "--coord_file", type=str, action='store',
            help="Please specify the file with coordinates", required=True)
        parser.add_argument("-r", "--report_file", type=str, action='store',
            help="Please specify the 'Report Territory Management' file", required=True)
        parser.add_argument("-o", "--output_file", type=str, action='store',
            help="Please specify the output file name", required=False)
        parser.add_argument("-s", "--samples_threshold", type=int, action='store',
            help="Specify a minimal samples number in a class", default=2, required=False)

        args = parser.parse_args()

        # set an output file name
        output_file = os.path.splitext(args.report_file)[0] + " Updated.xlsx" if args.output_file == None else args.output_file

        # Initialize
        tf = TerritoryFinder(args.coord_file, args.report_file, output_file, samples_threshold=args.samples_threshold)

        total_steps = 5

        logger.info(f"Step 1 of {total_steps}: Loading and prepare data")
        tf.load_data()

        logger.info(f"Step 2 of {total_steps}: Validate the model")
        tf.validate()

        logger.info(f"Step 3 of {total_steps}: Train the model")
        tf.fit()

        logger.info(f"Step 4 of {total_steps}: Prepare report")
        tf.get_report()

        logger.info(f"Step 5 of {total_steps}: Save report")
        tf.save_report()

        logger.info(f"Total elapsed time {time() - t0:.3f} sec.")
    
    except Exception as err:
        logger.exception(err)

    finally:
        sys.exit(1)

if __name__ == '__main__':
    main()