# Open the input and output files
with open('output.txt', 'r', encoding='utf-8') as input_file, open('output2.txt', 'w', encoding='utf-8') as output_file:
    # Loop through each line in the input file
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
            if len(current_line) + len(word) + 1 > 79:
                output_lines.append(current_line)
                current_line = word
            # Otherwise, add the next word plus a space to the current line
            else:
                current_line += ' ' + word
        # Add the final line to the output lines list
        output_lines.append(current_line)
        # Write each output line to the output file, followed by a newline character
        for output_line in output_lines:
            output_file.write(output_line + '\n')

exit()
# Open the input and output files
with open('output.txt', 'r', encoding='utf-8') as input_file, open('output2.txt', 'w', encoding='utf-8') as output_file:
    # Loop through each line in the input file
    for line in input_file:
        # Strip any leading/trailing whitespace from the line
        line = line.strip()
        # Split the line into chunks of 79 characters or less
        chunks = [line[i:i+79] for i in range(0, len(line), 79)]
        # Write each chunk to the output file, followed by a newline character
        for chunk in chunks:
            output_file.write(chunk + '\n')


exit()
with open('./output.txt', 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

output = ''
left_words = []
for line in lines:
    character_count = 0
    newline = ''

    if len(left_words) > 0:
        for word in left_words:
            if character_count > 79:
                break
            newline+= word + ' '
            character_count += len(word)

    line_characters = len(line)
    left_words = []
    if line_characters+character_count > 79:
        for word in line.split(' '):
            if character_count + len(word) < 79:
                newline += word+' '
                character_count += len(word)
            else:
                character_count = 100
                left_words.append(word) 
    else:
        newline = line
    output += newline + '\n'

with open('./output2.txt', 'w', encoding='utf-8') as fp:
    fp.write(output)