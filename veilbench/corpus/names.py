"""Name and place pools for several cultures. Plain data; the generator draws from them with a seeded RNG.

`TRAIN_CULTURES` are the pools the statistical detector may be trained on. The rest only appear in
the evaluation corpus, so the held-out split contains names the trained model has never seen.
"""

POOLS = {
    "en": {
        "first": "James Mary Robert Patricia John Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen Daniel Nancy Matthew Lisa Anthony Betty Mark Sandra Paul Ashley Steven Emily Andrew Donna Kenneth Michelle Joshua Carol Kevin Amanda Brian Melissa George Deborah Edward Stephanie Ronald Rebecca Timothy Laura Jason Sharon Jeffrey Cynthia Ryan Kathleen Jacob Amy Gary Shirley Nicholas Angela Eric Helen Jonathan Anna Stephen Brenda Larry Pamela Justin Nicole Scott Ruth Brandon Katherine Benjamin Samantha Samuel Christine Gregory Emma Alexander Catherine Patrick Debra Frank Rachel Raymond Carolyn Jack Janet Dennis Tyler Heather Aaron Diane Jerry Olivia Henry Julie Adam Joyce Douglas Victoria Nathan Kelly Peter Christina Zachary Lauren Kyle Joan Walter Evelyn Harold Judith Jeremy Megan Ethan Cheryl Carl Andrea Keith Hannah Roger Martha Gerald Jacqueline Christian Frances Terry Gloria Sean Ann Arthur Teresa Austin Kathryn Noah Sara Lawrence Janice Jesse Jean Joe Alice Bryan Madison Billy Doris Jordan Abigail Albert Julia Dylan Judy Bruce Grace Willie Denise Gabriel Amber Alan Marilyn Logan Danielle Wayne Theresa Ralph Sophia Roy Marie Eugene Diana Randy Brittany Vincent Natalie Russell Isabella Louis Charlotte Philip Rose Bobby Alexis Johnny Kayla".split(),
        "last": "Smith Johnson Williams Brown Jones Miller Davis Wilson Anderson Taylor Moore Jackson Martin Lee Thompson White Harris Clark Lewis Robinson Walker Young Allen King Wright Scott Hill Green Adams Nelson Baker Hall Campbell Mitchell Carter Roberts Phillips Evans Turner Parker Edwards Collins Stewart Morris Murphy Cook Rogers Morgan Cooper Peterson Bailey Reed Kelly Howard Cox Ward Richardson Watson Brooks Wood James Bennett Gray Hughes Price Sanders Myers Long Ross Foster Powell Jenkins Perry Russell Sullivan Bell Coleman Butler Henderson Barnes Fisher Simmons Jordan Patterson Alexander Hamilton Graham Reynolds Griffin Wallace West Cole Hayes Bryant Gibson Ellis Stevens Murray Ford Marshall Owens Harrison McDonald Woods Washington Kennedy Wells Henry Freeman Webb Tucker Burns Crawford Olson Simpson Porter Hunter Gordon Shaw Snyder Mason Dixon Hunt Hicks Holmes Palmer Wagner Black Robertson Boyd Rose Stone Fox Warren Mills Meyer Rice Daniels Ferguson Nichols Stephens Weaver Ryan Gardner Payne Grant Dunn Kelley Spencer Hawkins".split(),
    },
    "es": {
        "first": "Jose Antonio Manuel Francisco Juan Javier Carlos Miguel Alejandro Rafael Fernando Pablo Sergio Jorge Alberto Luis Pedro Angel Adrian Diego Raul Ivan Ruben Enrique Oscar Andres Ramon Vicente Santiago Marcos Mario Victor Alvaro Ignacio Guillermo Carmen Josefa Isabel Ana Dolores Pilar Rosa Cristina Lucia Marta Elena Paula Mercedes Concepcion Rocio Beatriz Raquel Inmaculada Montserrat Silvia Nuria Sonia Irene Alicia Esperanza Amparo Remedios Soledad Milagros".split(),
        "last": "Garcia Fernandez Gonzalez Rodriguez Lopez Martinez Sanchez Perez Gomez Jimenez Ruiz Hernandez Diaz Moreno Alvarez Munoz Romero Alonso Gutierrez Navarro Torres Dominguez Vazquez Ramos Gil Ramirez Serrano Blanco Molina Morales Suarez Ortega Delgado Castro Ortiz Rubio Marin Sanz Nunez Iglesias Medina Garrido Cortes Castillo Lozano Guerrero Cano Prieto Mendez Calvo Gallego Vidal Leon Marquez Herrera Pena Flores Cabrera Campos Vega Fuentes Carrasco Diez Reyes Caballero Nieto Aguilar Pascual Santana Herrero Lorenzo Hidalgo Montero Gimenez Ibanez Ferrer Duran Benitez Mora Vargas Arias Carmona Crespo Roman Pastor Soto Saez Velasco Moya Soler Parra Esteban Bravo Gallardo Rojas".split(),
        "accented": {"Jose": "José", "Garcia": "García", "Fernandez": "Fernández", "Gonzalez": "González", "Rodriguez": "Rodríguez", "Lopez": "López", "Martinez": "Martínez", "Sanchez": "Sánchez", "Perez": "Pérez", "Gomez": "Gómez", "Jimenez": "Jiménez", "Hernandez": "Hernández", "Diaz": "Díaz", "Alvarez": "Álvarez", "Munoz": "Muñoz", "Nunez": "Núñez", "Concepcion": "Concepción", "Ramirez": "Ramírez", "Marquez": "Márquez", "Angel": "Ángel", "Andres": "Andrés", "Ramon": "Ramón", "Victor": "Víctor", "Ruben": "Rubén", "Raul": "Raúl", "Lucia": "Lucía", "Rocio": "Rocío"},
    },
    "ph": {
        "first": "Maria Jose Juan Antonio Jonathan Mark Jayson Angelo Christian Joseph Michael John Jerome Rommel Reynaldo Rolando Ramon Ricardo Rodrigo Rafael Joselito Eduardo Fernando Carlos Marvin Kenneth Dennis Bryan Jomar Renato Teodoro Emmanuel Gabriel Cristina Ana Rosario Josefina Elena Carmela Gloria Corazon Imelda Teresita Marites Lourdes Evangeline Jocelyn Angelica Kristine Jennifer Michelle Mary Rhea Maricel Rowena Lorna Cherry Jasmine Princess Aileen Dolores Remedios Ligaya Minda".split(),
        "compound": ["Maria Cristina", "Ma. Theresa", "Juan Carlos", "Jose Maria", "Mary Grace", "Rose Ann", "John Paul", "Ana Marie", "Maria Lourdes", "Mary Jane", "Joseph Aaron"],
        "last": "Santos Reyes Cruz Bautista Ocampo Garcia Mendoza Torres Tomas Andrada Castillo Flores Villanueva Ramos Aquino Navarro Salazar Mercado Pascual Valdez Rivera Gonzales Aguilar Domingo Soriano Abad Cabrera Lacson Macapagal Magsaysay Aguinaldo Osmena Roxas Marcos Estrada Arroyo Binay Sotto Zubiri Escudero Gatchalian Tolentino Enriquez Manalo Dizon Padilla Cuevas Lim Tan Sy Ong Uy Chua Yap Dy Lao".split(),
        "particled": ["dela Cruz", "de los Santos", "del Rosario", "de Guzman", "dela Rosa", "de Leon", "de la Torre", "San Jose"],
    },
    "de": {
        "first": "Hans Peter Klaus Wolfgang Jurgen Dieter Uwe Stefan Thomas Michael Andreas Matthias Christian Markus Sebastian Florian Lukas Felix Jonas Anna Maria Ursula Monika Petra Sabine Susanne Claudia Birgit Katrin Julia Lena Sophie Hannah Laura Johanna".split(),
        "last": "Muller Schmidt Schneider Fischer Weber Meyer Wagner Becker Schulz Hoffmann Schafer Koch Bauer Richter Klein Wolf Schroder Neumann Schwarz Zimmermann Braun Kruger Hofmann Hartmann Lange Schmitt Werner Krause Meier Lehmann Schmid Schulze Maier Kohler Herrmann Konig Walter Mayer Huber Kaiser Fuchs Peters Lang Scholz Moller Weiss Jung Hahn Schubert Vogel Friedrich Keller Gunther Frank Berger Winkler Roth Beck Lorenz Baumann Franke Albrecht Schuster Simon Ludwig Bohm Winter Kraus Schumacher Kramer Vogt Stein Jager Otto Sommer Gross Seidel Heinrich Brandt Haas Schreiber Graf Schulte Dietrich Ziegler Kuhn Pohl Engel Horn Busch Bergmann Voigt Sauer Arnold Wolff Pfeiffer".split(),
        "accented": {"Muller": "Müller", "Jurgen": "Jürgen", "Schafer": "Schäfer", "Schroder": "Schröder", "Kruger": "Krüger", "Kohler": "Köhler", "Konig": "König", "Moller": "Möller", "Gunther": "Günther", "Bohm": "Böhm", "Jager": "Jäger", "Kuhn": "Kühn"},
    },
    "in": {
        "first": "Aarav Vivaan Aditya Vihaan Arjun Sai Reyansh Ayaan Krishna Ishaan Rohan Rahul Amit Sanjay Vijay Rajesh Suresh Ramesh Anil Sunil Deepak Manoj Ajay Vikram Karthik Naveen Pradeep Prakash Aanya Diya Saanvi Ananya Priya Pooja Neha Kavita Sunita Anita Meera Lakshmi Divya Shreya Ishita Riya Nandini Aishwarya Deepika Sneha".split(),
        "last": "Sharma Verma Gupta Singh Kumar Patel Shah Mehta Joshi Reddy Nair Iyer Menon Pillai Rao Naidu Chowdhury Banerjee Mukherjee Das Bose Ghosh Sen Desai Kulkarni Patil Jain Agarwal Bansal Kapoor Malhotra Khanna Chopra Saxena Mishra Tiwari Pandey Yadav Thakur Bhatt".split(),
    },
    "zh": {
        "first": "Wei Fang Na Min Jing Li Qiang Lei Yang Jun Tao Xin Hui Ming Hao Yu Chen Jie Ping Yan Xiu Ying Hong Lan Mei Ling Bo Chao Gang Kai Long Peng Rui Shan Ting Xiao Yi Zhen".split(),
        "last": "Wang Li Zhang Liu Chen Yang Huang Zhao Wu Zhou Xu Sun Ma Zhu Hu Guo He Lin Gao Luo Zheng Liang Xie Song Tang Han Feng Deng Cao Peng Zeng Xiao Tian Dong Pan Yuan Cai Jiang Du Ye Cheng Wei Su Lu Ding Ren Shen Yao Cui Zhong Tan Fan".split(),
    },
    "ar": {
        "first": "Mohammed Ahmed Ali Omar Hassan Hussein Ibrahim Yusuf Khalid Mahmoud Mustafa Karim Tariq Samir Nabil Faisal Rashid Walid Fatima Aisha Khadija Zainab Mariam Layla Noor Salma Huda Amal Rania Samira Nadia Yasmin Dina".split(),
        "last": "Haddad Khalil Nasser Mansour Saleh Farah Aziz Qureshi Rahman Sayegh Jabbar Hamdan Suleiman Darwish Habib Shaheen Awad Bakr Zaman Malik Rashid Tahir Sabbagh Khoury Najjar Hakim".split(),
    },
    "vn": {
        "first": "Anh Binh Cuong Dung Duc Hai Hieu Hung Khanh Linh Long Minh Nam Phuc Quang Son Thanh Tuan Vinh Lan Huong Mai Ngoc Phuong Thao Trang Yen Hoa Thu Lien".split(),
        "last": "Nguyen Tran Le Pham Hoang Huynh Phan Vu Vo Dang Bui Do Ho Ngo Duong Ly".split(),
    },
    "ng": {
        "first": "Chinedu Emeka Ifeanyi Obinna Tunde Segun Kunle Femi Olumide Babatunde Adebayo Chidi Nnamdi Uche Ngozi Chioma Amaka Funke Folake Bukola Yetunde Adaeze Ifeoma Nkechi Titilayo".split(),
        "last": "Okafor Adeyemi Balogun Okonkwo Eze Nwosu Obi Ibrahim Abubakar Bello Lawal Adeleke Ogunbanjo Ogunleye Oladipo Afolabi Chukwu Nnadi Onyeka Olawale Fashola Akinola Adebisi Oyelaran".split(),
    },
}

TRAIN_CULTURES = ["en", "es", "ph", "de"]
ALL_CULTURES = list(POOLS)

STREETS_EN = "Oak Maple Cedar Pine Elm Washington Lake Hill Park Sunset Highland Madison Jefferson Lincoln Riverside Church Spring Willow Chestnut Walnut Dogwood Forest Meadow Ridge Valley".split()
STREETS_PH = "Mabini Rizal Bonifacio Quezon Osmena Magallanes Colon Jakosalem Lahug Banilad Mango Escario Sanciangko Burgos Luna Luzuriaga Zamora Aguinaldo Jacinto Roxas Laurel Katipunan Kalayaan Maharlika Sampaguita Ilang-ilang".split()
STREETS_ES = "Alcala Mayor Atocha Serrano Velazquez Goya Princesa Toledo Segovia Fuencarral Sol Preciados Arenal Recoletos Castellana Bailen Embajadores Delicias Lavapies".split()
CITIES_PH = [("Cebu City", "6000"), ("Mandaue City", "6014"), ("Lapu-Lapu City", "6015"), ("Quezon City", "1100"), ("Makati City", "1200"), ("Davao City", "8000"), ("Iloilo City", "5000"), ("Baguio City", "2600"), ("Cagayan de Oro", "9000"), ("Pasig City", "1600")]
CITIES_US = [("San Francisco", "CA", "94103"), ("Austin", "TX", "78701"), ("Denver", "CO", "80202"), ("Seattle", "WA", "98101"), ("Boston", "MA", "02108"), ("Portland", "OR", "97201"), ("Chicago", "IL", "60601"), ("Atlanta", "GA", "30303")]
CITIES_ES = [("Madrid", "28014"), ("Barcelona", "08001"), ("Sevilla", "41001"), ("Valencia", "46001"), ("Bilbao", "48001"), ("Zaragoza", "50001")]
