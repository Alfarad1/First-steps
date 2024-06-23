import string 
def prepare(review: str):
    res = review
    res = res.lower()
    res = res.translate(str.maketrans('','',string.punctuation))
    res = res.translate(str.maketrans('','','1234567890'))
    return res