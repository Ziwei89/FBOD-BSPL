class person(object):
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
        self.old_class = None

    @property
    def name(self):
        return self.__name
    @property
    def age(self):
        return self.__age

if __name__ == "__main__":
    person_list = []

    person_list.append(person("aa", 15))
    person_list.append(person("bb", 86))
    person_list.append(person("cc", 39))
    person_list.append(person("dd", 28))
    person_list.append(person("ee", 99))

    print("remove before:")
    for aperson in person_list:
        print(aperson.name)

    for aperson in person_list:
        if aperson.age > 80:
            person_list.remove(aperson)
    
    print("remove after:")
    for aperson in person_list:
        print(aperson.name)