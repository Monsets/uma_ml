import cv2
import numpy as np
import telebot as tb

from classifier import Classifier

TOKEN = '950107247:AAFy40ZucUvIdrfXqWR1HVnAwQcK8KY-_hs'


def classify(id):
    file_info = bot.get_file(id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    img = cv2.imread('image.jpg')
    img = cv2.resize(img, (224, 224))
    img = np.array(img).reshape(1, 224, 224, 3)

    return cl.predict(img)


bot = tb.TeleBot(TOKEN)
cl = Classifier()


@bot.message_handler(content_types=['document', 'photo'])
def get_text_messages(message):
    if message.photo:
        for p in message.photo:
            bot.send_message(message.from_user.id, classify(p.file_id))
    if message.document:
        for d in message.document:
            bot.send_message(message.from_user.id, classify(d.file_id))


if __name__ == '__main__':
    print('starting...')
    bot.polling()
