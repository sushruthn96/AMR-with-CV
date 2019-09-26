import urllib2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SaveImage(object):
    name = ""
    index = 1

    def __init__(self, name):
        self.name = name  # instance variable unique to each instance

    def save(self, img_url):
        img_file_name = self.name + "%03d" % self.index + '.jpg'
        try:
            self._download_img(img_url, img_file_name)
            self.index += 1
        except:
            print("Cannot dload image " + img_url)

    @staticmethod
    def _download_img(image_url, file_name):
        request = urllib2.Request(image_url)
        img = urllib2.urlopen(request).read()
        with open(file_name, 'wb') as img_file:
            img_file.write(img)

# saveObj = SaveJSON("analog meter")
# img_url = "https://burst.shopifycdn.com/photos/white-faced-watch_925x@2x.jpg"
# img_src = "https://burst.shopify.com/photos/white-faced-watch"
# tags = [1,2,3,4]
# saveObj.save(img_src, img_url , tags)