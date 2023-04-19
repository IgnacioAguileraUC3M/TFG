class normalizer_class:
    def __init__(self):
        pass
    def normalize(self, max_words, input_file):
        output_file = ''
        for line in input_file:
            # Strip any leading/trailing whitespace from the line
            line = line.strip()
            # Initialize a list to hold the output lines
            output_lines = []
            # Split the line into words
            words = line.split()
            # Initialize a variable to hold the current line
            current_line = ''
            # Loop through each word in the line
            for word in words:
                # If the current line plus the next word plus a space is longer than 79 characters,
                # add the current line to the output lines list and start a new line with the current word
                if len(current_line) + len(word) + 1 > max_words:
                    output_lines.append(current_line)
                    current_line = word
                # Otherwise, add the next word plus a space to the current line
                else:
                    current_line += ' ' + word
            # Add the final line to the output lines list
            output_lines.append(current_line)
            # Write each output line to the output file, followed by a newline character
            for output_line in output_lines:
                output_file+=output_line+'\n'
        return output_file