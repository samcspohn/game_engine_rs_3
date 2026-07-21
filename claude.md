update Readme.md with current implementation details if there are any significant changes

avoid fallbacks
if X doesn't work dont create code Y to silently fail. this appears as a bug to me but correct to you. a crash is easier to debug and identify. only implement fallbacks if instructed

avoid overly verbose comments. prefer to make code self documenting. if a comment is needed, make it concise and to the point
