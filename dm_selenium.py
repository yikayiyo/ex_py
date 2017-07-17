from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#Chrome("/usr/lib/chromium-browser/chromedriver")
driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
driver.get("http://www.python.org")
assert "Python" in driver.title
elem = driver.find_element_by_name("q")
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)

# driver.get("http://music.163.com/#/song?id=410802751")
print driver.page_source