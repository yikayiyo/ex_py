from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains

#Chrome("/usr/lib/chromium-browser/chromedriver")
browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
try:
    # browser.get("https://www.baidu.com")
    #two forms
    # elem = browser.find_element_by_id("kw")
    # elem = browser.find_element(By.ID,"kw")
    # elem.send_keys("Python")
    # elem.send_keys(Keys.ENTER)

    # driver.get("http://music.163.com/#/song?id=410802751")
    # print browser.current_url
    # print browser.get_cookies()
    # print browser.page_source
    url="http://www.runoob.com/try/try.php?filename=jqueryui-api-droppable"
    browser.get(url)
    # browser.switch_to_frame('iframeResult')
    # source=browser.find_element_by_css_selector('#draggable')
    # target=browser.find_element_by_css_selector('#droppable')
    # actions=ActionChains(browser)
    # actions.drag_and_drop(source,target)
    # actions.perform()
    browser.execute_script('window.open()')
    browser.switch_to.window(browser.window_handles[0])
finally:
#     browser.close()
    pass