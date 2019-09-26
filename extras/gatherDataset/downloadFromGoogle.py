from imagesoup import ImageSoup
import SaveImage

soup = ImageSoup()

# query = raw_input("Enter the query: ")
# number_of_urls = int(input("Enter the number of urls to fetch: "))

number_of_urls = 500
queries = ["digital_meter"]

for query in queries:
    saveObj = SaveImage.SaveImage("./gImages/meters2/" + query + "/" + query)
    images = soup.search(query, n_images=number_of_urls)
    for i in range(0, len(images)):
        img_url = images[i].URL
        print (i, img_url)
        saveObj.save(img_url)

