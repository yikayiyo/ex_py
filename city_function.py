#-*- ecoding:utf-8 -*-

# def city_country(city,country):
#     '''返回 City,Country 字符串'''
#     return city+','+country

def city_country(city,country,population=''):
    '''返回 City,Country 字符串'''
    if population:
        return city+','+country+' - '+population[0:10]+' '+population[11:]
    else:
        return city+','+country