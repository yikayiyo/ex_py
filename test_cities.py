import unittest
from city_function import city_country

class CityTestCase(unittest.TestCase):
    def test_city_country(self):
        str=city_country('shanghai','china')
        self.assertEqual(str, 'shanghai,china')

    def test_city_country_population(self):
        str=city_country('shanghai','china','population=10000')
        self.assertEqual(str, 'shanghai,china - population 10000')


if __name__ == '__main__':
    unittest.main()
