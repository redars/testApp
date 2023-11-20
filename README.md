# Комментарии тестов
#Во всех папках с тестами#: 
CityGrid_1 - изображение сгенерированного города, где красные блоки - obstructed, зеленые - нет. 
NetworkCoverage_1 - изображение покрытия сети после генерации города, все блоки красные т.к. вышек пока не установлено. CityGrid_2 - изображение города с поставленными вышками, добавляются желтые блоки с числами внутри. Желтые блоки - это вышки, числа в них - дистация действия вышки. NetworkCoverage_2 - изображение покрытия сети после установки вышек. output - текстовый файл, содержащий  результат выполнеия программы в консоль.
test_1: Размер города - 5 на 5. 40% obstructed блоков. Вышки расставлялись рандомно (10 попыток поставить вышку).
test_2: Размер города - 5 на 5. 40% obstructed блоков. Вышки расставлялись по принципу минимизации количества вышек для покрытия всей территории.
test_3: Размер города - 20 на 20. 50% obstructed блоков. Вышки расставлялись рандомно (40 попыток поставить вышку).
test_4: Размер города - 10 на 20. 50% obstructed блоков. Вышки расставлялись рандомно (40 попыток поставить вышку).
