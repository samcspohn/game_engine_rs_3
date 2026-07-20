update Readme.md with current implementation details if there are any significant changes

avoid fallbacks
if X doesn't work dont create code Y to silently fail. this appears as a bug to me but correct to you. a crash is easier to debug and identify. only implement fallbacks if instructed
