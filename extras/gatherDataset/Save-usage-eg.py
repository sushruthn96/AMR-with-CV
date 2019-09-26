#Run Save.py to get Save.pyc

import SaveImage

saveObj = SaveImage.SaveImage("analog meter")
img_url = "https://burst.shopifycdn.com/photos/white-faced-watch_925x@2x.jpg"
saveObj.save(img_url)
