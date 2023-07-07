from bs4 import BeautifulSoup

html_doc = open("C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/AI-For-Webdevlopment/templates/index.html")

soup = BeautifulSoup(html_doc)
print(soup)
new_h1 = soup.new_tag("h2")

new_h1.string = "Hello one"

#If thier is one tag only then can use below code
tag = soup.find("h1")
tag.insert_after(new_h1)
#tag.insert_before(new_h1)
value = soup.find("body")

#If there are same number of tags and want to select one of them
tag = soup.find_all("h1")
tag[1].insert_after(new_h1)
value = soup.find("body")

#Replace the tag
old_tag = soup.find_all("h2")
new_tag = soup.new_tag("p")
new_tag.string = "Hi! Everyone"

old_tag[1].replace_with(new_tag)

savechange = soup.prettify("utf-8")
with open("C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/AI-For-Webdevlopment/templates/index.html", "wb") as file:
    file.write(savechange)