# Лабораторная работа №1
## Цель работы
Получить навыки разработки CI/CD pipeline для ML моделей с достижением
метрик моделей и качества.

## Ход работы
1. Создать репозитории модели на GitHub, регулярно проводить commit + push в ветку разработки, важна история коммитов;
2. Провести подготовку данных для набора данных, согласно варианту задания;
3. Разработать ML модель с ЛЮБЫМ классическим алгоритмом классификации, кластеризации, регрессии и т. д.;
4. Конвертироватьмодельиз*.ipynbв.pyскрипты,реализоватьAPIсервис с методом на вывод модели, фронтальная часть по желанию;
5. Покрыть код тестами, используя любой фреймворк/библиотеку;
6. Задействовать DVC;
7. Использовать Docker для создания docker image.
8. Наполнить дистрибутив конфигурационными файлами:
• config.ini: гиперпараметры модели;
• Dockerfile и docker-compose.yml: конфигурация создания
контейнера и образа модели; 1
• requirements.txt: используемые зависимости (библиотеки) и их версии;
• dev_sec_ops.yml: подписи docker образа, хэш последних 5 коммитов в репозитории модели, степень покрытия тестами (необязательно);
• scenario.json: сценарии тестирования запущенного контейнера модели (необязательно).
9. СоздатьCIpipeline(Jenkins,TeamCity,CircleCIидр.)длясборкиdocker image и отправки его на DockerHub, сборка должна автоматически стартовать по pull request в основную ветку репозитория модели;
10.Создать CD pipeline для запуска контейнера и проведения функционального тестирования по сценарию, запуск должен стартовать по требованию или расписанию или как вызов с последнего этапа CI pipeline;
11.Результаты функционального тестирования и скрипты конфигурации CI/CD pipeline приложить к отчёту.
Результаты работы:
1. Отчёт о проделанной работе;
2. Ссылка на репозиторий GitHub;
3. Ссылка на docker image в DockerHub;
4. Актуальный дистрибутив модели в zip архиве.
Обязательно обернуть модель в контейнер (этап CI) и запустить тесты внутри контейнера (этап CD).



### GitHub
https://github.com/ka1mar/MLOps-lab1
### Docker image
https://hub.docker.com/repository/docker/ka1mar/mlops-lab1
