## Urban Dictionary API

Username: mt11731
Password: normalization
URL: https://market.mashape.com/community/urban-dictionary#

Example request in Python to search for "lmap":
```
response = unirest.get("https://mashape-community-urban-dictionary.p.mashape.com/define?term=lmap",
  headers={
    "X-Mashape-Key": "kGgb1BGK61mshaURCbIxwNkOFnpep1p4jw5jsn8aUG3XWq4ihm",
    "Accept": "text/plain"
  }
)
```
