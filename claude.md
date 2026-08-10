update Readme.md with current implementation details if there are any significant changes

follow YAGNI principles, and prefer one-liner solutions

avoid fallbacks as crutch for a new feature not working especially in the case of one implementation replacing another. do not leave the previous impl in place with the new impl silently falling back to it.

do implement fallbacks in the case of user friendliness. the program shouldn't crash in the case a user drag and drops an incorrect value to a drop zone. 

avoid overly verbose comments. prefer to make code self documenting. if a comment is needed, make it concise and to the point. prefer 1 line comments up to 3 lines

once completed, explain your changes and how they work
