import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from collections import deque

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, key, value):
        new_node = Node(key, value)
        new_node.next = self.head
        self.head = new_node

    def search(self, key):
        current = self.head
        while current:
            if current.key == key:
                return current.value
            current = current.next
        return None

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [LinkedList() for _ in range(size)]

    def hash_function(self, key):
        total = 0
        for char in key:
            total += ord(char)
        return total % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        self.table[index].insert(key, value)

    def search(self, key):
        index = self.hash_function(key)
        return self.table[index].search(key)

class InterestHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [LinkedList() for _ in range(size)]

    def hash_function(self, key):
        total = 0
        for char in key:
            total += ord(char)
        return total % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        self.table[index].insert(key, value)

    def search(self, key):
        index = self.hash_function(key)
        return self.table[index].search(key)

class UserGraph:
    def __init__(self, user_table, interest_table):
        self.user_table = user_table
        self.interest_table = interest_table
        self.graph = nx.DiGraph()

    def add_relationships(self):
        for linked_list in self.user_table.table:
            current_node = linked_list.head
            while current_node:
                user = current_node.value
                following_list = user["following"]
                followers_list = user["followers"]

                for follower_username in followers_list:
                    self.graph.add_edge(follower_username, user["username"])

                for following_username in following_list:
                    self.graph.add_edge(user["username"], following_username)

                current_node = current_node.next

    def determine_interest_areas(self):
        stop_words = set(stopwords.words('english'))
        for linked_list in self.user_table.table:
            current_node = linked_list.head
            while current_node:
                user = current_node.value
                tweets = user["tweets"]

                tweets_text = ' '.join(tweets) if isinstance(tweets, list) else tweets

                tokens = word_tokenize(tweets_text.lower())

                filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

                freq_dist = FreqDist(filtered_tokens)

                user["interest_areas"] = [word for word, freq in freq_dist.items() if freq >= 1]

                for interest in user["interest_areas"]:
                    interest_users = self.interest_table.search(interest)
                    if interest_users:
                        interest_users.append(user["username"])
                    else:
                        self.interest_table.insert(interest, [user["username"]])

                current_node = current_node.next

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', connectionstyle='arc3,rad=0.1')
        plt.title("User Graph")
        plt.show()

    def write_relationships_to_file(self, filename=r'C:\Users\user\Desktop\yazdir.txt'):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("Followers-followed Relationship of Users\n")
            for linked_list in self.user_table.table:
                current_node = linked_list.head
                while current_node:
                    user = current_node.value
                    username = user["username"]
                    following_list = user["following"]
                    followers_list = user["followers"]

                    file.write(f"User: {username}\n")
                    file.write("Followers:\n")
                    for follower_username in followers_list:
                        file.write(f" - {follower_username}\n")

                    file.write("Following:\n")
                    for following_username in following_list:
                        file.write(f" - {following_username}\n")

                    file.write("\n")
                    current_node = current_node.next

    def write_common_interests_to_file(self, filename=r'C:\Users\user\Desktop\yazdir.txt'):
        with open(filename, 'a', encoding='utf-8') as file:
            for linked_list in self.interest_table.table:
                current_node = linked_list.head
                while current_node:
                    interest = current_node.key
                    users = current_node.value
                    if len(users) > 1:
                        common_users = []
                        for username in users:
                            user_interests = self.user_table.search(username)["interest_areas"]
                            if interest in user_interests:
                                common_users.append(username)

                        if len(common_users) > 1:
                            file.write(f"Interest: {interest}, \nUsers: {', '.join(common_users)}\n")
                            for username in common_users:
                                user_info = self.user_table.search(username)

                    current_node = current_node.next
    #DFS
    def dfs_keywords_in_tweets(self, start_user):
        keyword_input = input("Aranacak anahtar kelimeleri ',' ile ayırarak girin: ")
        keywords = [keyword.strip() for keyword in keyword_input.split(",")]

        visited = set()
        stack = [start_user]

        print(f"\nDFS: Searching for {keywords} in tweets starting from {start_user}:")

        while stack:
            current_user = stack.pop()
            if current_user not in visited:
                user_info = self.user_table.search(current_user)
                if user_info:
                    tweets = user_info["tweets"]
                    if isinstance(tweets, list):
                        for tweet in tweets:
                            if any(keyword.lower() in tweet.lower() for keyword in keywords):
                                print(f"{current_user}: {tweet}\n")

                    visited.add(current_user)
                    stack.extend(self.graph.neighbors(current_user))

    # BFS
    def bfs_common_interests(self, user1, user2):
        visited = set()
        queue = deque([(user1, [])])

        print(f"BFS: Finding common interests between {user1} and {user2}:")

        while queue:
            current_user, path = queue.popleft()
            if current_user not in visited:
                user_info = self.user_table.search(current_user)
                if user_info:
                    interests = set(user_info.get("interest_areas", []))
                    if user2 in interests:
                        print(f"Common interest found: {user2} between {user1} and {current_user}, Common interests: {', '.join(interests)}")
                        return

                    visited.add(current_user)

                    for neighbor in set(self.graph.neighbors(current_user)) - visited:
                        queue.append((neighbor, path + [current_user]))
#oneuser
    def draw_user_graph(self, username):
        user_info = self.user_table.search(username)
        if user_info:
            user_graph = nx.DiGraph()

            followers_list = user_info["followers"]
            for follower_username in followers_list:
                user_graph.add_edge(follower_username, username)

            following_list = user_info["following"]
            for following_username in following_list:
                user_graph.add_edge(username, following_username)

            pos = nx.spring_layout(user_graph)
            nx.draw(user_graph, pos, with_labels=True, font_weight='bold', connectionstyle='arc3,rad=0.1')
            plt.title(f"{username} Kullanıcısının Grafiği")
            plt.show()
        else:
            print(f"'{username}' kullanıcısı bulunamadı.")



# Ana kodunuz
data = pd.read_json(r'C:\Users\user\Desktop\twitter.json')

userDataTable = HashTable(size=10)
interestDataTable = InterestHashTable(size=10)

userGraph = UserGraph(userDataTable, interestDataTable)

for index, row in data.iterrows():
    theUser = {
        "username": row["username"],
        "name": row["name"],
        "followers_count": row["followers_count"],
        "following_count": row["following_count"],
        "language": row["language"],
        "region": row["region"],
        "tweets": row["tweets"],
        "following": row["following"],
        "followers": row["followers"],
        "interest_areas": row.get("interest_areas", [])
    }
    userDataTable.insert(row["username"], theUser)

userGraph.add_relationships()
userGraph.write_relationships_to_file()
userGraph.determine_interest_areas()
userGraph.write_common_interests_to_file()

# Menu
import sys

while True:
    def menu(secim):
        if secim == 1:
            girilen_kullanici_adi = input("Çizim yapılacak kullanıcı adını girin: ")
            userGraph.draw_user_graph(girilen_kullanici_adi)
        elif secim == 2:
            userGraph.draw_graph()
        elif secim == 3:
            kad = input("Baslangıc Kullanisini gir:")
            userGraph.dfs_keywords_in_tweets(kad)
        elif secim == 4:
            u1 = input("İlk kullanıcıyı girin: ")
            u2 = input("İkinci kullanıcıyı girin: ")
            userGraph.bfs_common_interests(u1, u2)
        elif secim == 5:
            print('Çıkış yapıldı')
            sys.exit()
        else:
            print('Geçersiz Seçim')

    print('----- MENÜ -----')
    print('1. Belirli bir Kişinin Grafini Çiz')
    print('2. Tüm Grafları Çiz')
    print('3.Anahtar Kelimeye Göre DFS ile Tweet Bul')
    print('4.BFS ile iki Kullancı Arasında Bağlantı Bul')
    print('5. Çıkış Yap')


    kullanici_secimi = input("Yapmak istediğiniz işlemi seçiniz: ")
    try:
        kullanici_secimi = int(kullanici_secimi)
        menu(kullanici_secimi)
    except ValueError:
        print("Lütfen geçerli bir sayı girin.")






