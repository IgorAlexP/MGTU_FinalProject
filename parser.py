import requests
import sqlite3
from bs4 import BeautifulSoup

# URL Центрального Банка для курса валют
url = 'https://www.cbr.ru/currency_base/daily/'

# Создаем базу данных SQLite
conn = sqlite3.connect('currency_exchange.db')
cursor = conn.cursor()

# Создаем таблицу для хранения данных
cursor.execute('''
    CREATE TABLE IF NOT EXISTS currency_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        currency_name TEXT,
        currency_code TEXT,
        exchange_rate REAL
    )
''')

def update_currency_data():
    # Отправляем GET-запрос и получаем HTML-страницу
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Находим таблицу с курсами валют
        table = soup.find('table', {'class': 'data'})

        # Извлекаем данные и сохраняем их в базу данных
        rows = table.find_all('tr')
        for row in rows[1:]:  # Пропустить заголовок таблицы
            columns = row.find_all('td')
            currency_name = columns[3].text
            currency_code = columns[1].text
            exchange_rate = float(columns[4].text.replace(',', '.'))

            # Вставляем данные в базу данных
            cursor.execute('''
                INSERT INTO currency_rates (currency_name, currency_code, exchange_rate)
                VALUES (?, ?, ?)
            ''', (currency_name, currency_code, exchange_rate))

        # Сохраняем изменения и закрываем соединение с базой данных
        conn.commit()
    else:
        print('Не удалось получить данные с Центрального Банка')

def view_currency_data():
    cursor.execute('SELECT * FROM currency_rates')
    data = cursor.fetchall()
    for row in data:
        print(f'ID: {row[0]}, Название валюты: {row[1]}, Код валюты: {row[2]}, Курс: {row[3]}')

# Функция для обновления данных и просмотра базы данных
def main():
    update_currency_data()
    view_currency_data()

if __name__ == '__main__':
    main()

# Закрываем соединение с базой данных
conn.close()
