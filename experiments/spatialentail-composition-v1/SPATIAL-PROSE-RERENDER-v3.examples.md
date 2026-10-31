# Spatial-Prose Training Examples

Actual v3 references for unchanged v2 questions. Both styles use the same checked evidence.

## direction: A-D (first)

ID: `controlled-train-v2-q00000-a0-direction-m0-unique-entailed`

Consider a map with multiple objects:
Florist is to the South of Bookstore. Gas Station is to the South of Bookstore. Florist is to the South of Bank. Bank is to the South of Bookstore. Gym is to the North of Bookstore. Bookstore is to the North of Supermarket.

In which direction is Gym relative to Bookstore?
A. Northeast
B. Southwest
C. North
D. Southeast
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: Gym is to the North of Bookstore.

P5 places Gym north of Bookstore.

North is option C.

Answer: C

### Symbolic

Premises used (labels follow their order in the question):
P5: Gym is to the North of Bookstore.

[P5] N(Gym,Bookstore).

North is option C.

Answer: C

## direction: unique-unlisted (first)

ID: `controlled-train-v2-q00096-a0-direction-m1-unique-entailed`

Consider a map with multiple objects:
Hospital is to the Southeast of Library. Library is to the South of Train Station. Stadium is to the Northwest of Theater. Park is to the Southwest of Library. High School is to the Southeast of Library. University is to the Southeast of Library. Library is to the Northwest of Gas Station. Library is to the Northwest of Stadium. Hotel is to the Southwest of Library. Theater is to the Southeast of Library. Cafe is to the Southeast of Library. Library is to the Northeast of Cinema.

In which direction is Train Station relative to Library?
A. South
B. East
C. Southeast
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P2: Library is to the South of Train Station.

P2 places Train Station north of Library.

North is not listed among A-D.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P2: Library is to the South of Train Station.

[P2] N(Train Station,Library).

North is not listed among A-D.

Answer: F

## direction: A-D (L8)

ID: `controlled-train-v2-q01134-a0-direction-m0-unique-entailed`

Consider a map with multiple objects:
Theater is to the Southeast of Restaurant. Cafe is to the Northwest of Museum. Park is to the Southeast of Gas Station. Cafe is to the West of High School. High School is to the East of Stadium. Cafe is to the Northwest of Bookstore. Grocery Store is to the Southeast of Fire Station. Florist is to the Southeast of Restaurant. Grocery Store is to the Northwest of Police Station. Florist is to the East of Courthouse. Police Station is to the Northwest of Gas Station. Cafe is to the Northwest of Hotel. Veterinary Clinic is to the Northwest of Train Station. City Hall is to the Southeast of Cafe. Gym is to the Northeast of Cafe. Library is to the Southeast of Bookstore. Library is to the Southeast of Cafe. Bookstore is to the Southwest of Courthouse. Restaurant is to the Northeast of Bookstore. Stadium is to the Southwest of Fire Station. Train Station is to the Northwest of Cafe. Restaurant is to the Northwest of Courthouse. Veterinary Clinic is to the Southeast of Park.

In which direction is Stadium relative to Cafe?
A. Southwest
B. West
C. Southeast
D. North
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P3: Park is to the Southeast of Gas Station.
P4: Cafe is to the West of High School.
P5: High School is to the East of Stadium.
P7: Grocery Store is to the Southeast of Fire Station.
P9: Grocery Store is to the Northwest of Police Station.
P11: Police Station is to the Northwest of Gas Station.
P13: Veterinary Clinic is to the Northwest of Train Station.
P20: Stadium is to the Southwest of Fire Station.
P21: Train Station is to the Northwest of Cafe.
P23: Veterinary Clinic is to the Southeast of Park.

From west to east: Stadium, Fire Station, Grocery Store, Police Station, Gas Station, Park, Veterinary Clinic, Train Station and Cafe (P20, P7, P9, P11, P3, P23, P13, P21).
Stadium, High School and Cafe lie on the same horizontal line (P5, P4).
Together, these relations place Stadium west of Cafe.

West is option B.

Answer: B

### Symbolic

Premises used (labels follow their order in the question):
P3: Park is to the Southeast of Gas Station.
P4: Cafe is to the West of High School.
P5: High School is to the East of Stadium.
P7: Grocery Store is to the Southeast of Fire Station.
P9: Grocery Store is to the Northwest of Police Station.
P11: Police Station is to the Northwest of Gas Station.
P13: Veterinary Clinic is to the Northwest of Train Station.
P20: Stadium is to the Southwest of Fire Station.
P21: Train Station is to the Northwest of Cafe.
P23: Veterinary Clinic is to the Southeast of Park.

[P20, P7, P9, P11, P3, P23, P13, P21] x(Stadium) < x(Fire Station) < x(Grocery Store) < x(Police Station) < x(Gas Station) < x(Park) < x(Veterinary Clinic) < x(Train Station) < x(Cafe).
[P5, P4] y(Stadium) = y(High School) = y(Cafe).
Therefore W(Stadium,Cafe).

West is option B.

Answer: B

## direction: no-entailed-answer (first)

ID: `controlled-train-v2-q01400-a0-direction-m0-unique-entailed`

Consider a map with multiple objects:
Stadium is to the North of Veterinary Clinic. Stadium is to the North of Bakery. Stadium is to the Northeast of Museum. Veterinary Clinic is to the South of Post Office. Museum is to the Southwest of City Hall. Stadium is to the North of Post Office.

In which direction is City Hall relative to Stadium?
A. Southeast
B. East
C. Northeast
D. North
E. Cannot be determined
F. None of the above

### Natural

Case 1: One arrangement satisfying the premises is as follows. From west to east, the order is Museum, City Hall and Stadium. From south to north, the order is Museum, City Hall and Stadium. So City Hall is southwest of Stadium.

Case 2: One arrangement satisfying the premises is as follows. Museum lies to the left of City Hall and Stadium. City Hall and Stadium share the same east-west position. From south to north, the order is Museum, Stadium and City Hall. So City Hall is north of Stadium.

The same premises allow different outcomes, so no direction is guaranteed.

Answer: E

### Symbolic

Case 1 (satisfies the premises): x(Museum) < x(City Hall) < x(Stadium). y(Museum) < y(City Hall) < y(Stadium). => SW(City Hall,Stadium).

Case 2 (satisfies the premises): x(Museum) < x(City Hall) = x(Stadium). y(Museum) < y(Stadium) < y(City Hall). => N(City Hall,Stadium).

The same premises allow different outcomes, so no direction is guaranteed.

Answer: E

## direction: inconsistent (first)

ID: `controlled-train-v2-q01850-a0-direction-m0-unique-entailed`

Consider a map with multiple objects:
Library is to the West of Courthouse. Library is to the East of Police Station. Clinic is to the West of Library. Library is to the East of Pharmacy. Library is to the East of Stadium. Courthouse is to the West of Library.

In which direction is Courthouse relative to Library?
A. South
B. Southeast
C. East
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Library is to the West of Courthouse.
P6: Courthouse is to the West of Library.

From west to east: Library, Courthouse and Library (P1, P6).
This would put Library further west than itself, which is impossible.

The premises contradict one another.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P1: Library is to the West of Courthouse.
P6: Courthouse is to the West of Library.

[P1, P6] x(Library) < x(Courthouse) < x(Library).
A strict ordering cannot return to its starting point.

The premises contradict one another.

Answer: E

## which: A-D (first)

ID: `controlled-train-v2-q02000-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
University is to the South of Fire Station. Fire Station is to the North of Hotel. Hotel is to the North of Courthouse. Fire Station is to the South of Police Station. Fire Station is to the North of Train Station. Courthouse is to the South of Fire Station.

Which object is north of Fire Station?
A. Police Station
B. University
C. Courthouse
D. Hotel
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P4: Fire Station is to the South of Police Station.

P4 places Police Station north of Fire Station.

Case 1: One arrangement satisfying the premises is as follows. Courthouse, Hotel, Train Station and University are south of Fire Station. Police Station is north of Fire Station. Only Police Station qualifies in this arrangement.

Courthouse, Hotel, Train Station and University are not north of Fire Station in this arrangement, so none is guaranteed to be.

Only Police Station is guaranteed to be north of Fire Station.

Police Station is option A.

Answer: A

### Symbolic

Premises used (labels follow their order in the question):
P4: Fire Station is to the South of Police Station.

[P4] N(Police Station,Fire Station).

Case 1 (satisfies the premises): S(Courthouse,Fire Station) & S(Hotel,Fire Station) & S(Train Station,Fire Station) & S(University,Fire Station). N(Police Station,Fire Station). Qualifying objects = {Police Station}.

Case 1 falsifies the queried relation for {Courthouse, Hotel, Train Station, University}; none is entailed.

Only Police Station is guaranteed to be north of Fire Station.

Police Station is option A.

Answer: A

## which: unique-unlisted (first)

ID: `controlled-train-v2-q02072-a0-which-m1-unique-entailed`

Consider a map with multiple objects:
Gallery is to the Southeast of Gym. Gym is to the Northwest of Museum. Veterinary Clinic is to the Southeast of Gym. Gym is to the Northeast of Hotel. Gym is to the Northwest of Stadium. Gym is to the Northeast of Florist. Hospital is to the South of Gym. Gallery is to the Southeast of Hospital. Post Office is to the Southwest of Gym. Restaurant is to the Southwest of Gym. University is to the Southwest of Gym. High School is to the North of Gym.

Which object is north of Gym?
A. Hotel
B. Veterinary Clinic
C. University
D. Hospital
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P12: High School is to the North of Gym.

P12 places High School north of Gym.

Case 1: One arrangement satisfying the premises is as follows. Florist, Hotel, Post Office, Restaurant and University are southwest of Gym. Gallery, Museum, Stadium and Veterinary Clinic are southeast of Gym. High School is north of Gym. Hospital is south of Gym. Only High School qualifies in this arrangement.

Florist, Gallery, Hospital, Hotel, Museum, Post Office, Restaurant, Stadium, University and Veterinary Clinic are not north of Gym in this arrangement, so none is guaranteed to be.

Only High School is guaranteed to be north of Gym.

High School is not listed among A-D.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P12: High School is to the North of Gym.

[P12] N(High School,Gym).

Case 1 (satisfies the premises): SW(Florist,Gym) & SW(Hotel,Gym) & SW(Post Office,Gym) & SW(Restaurant,Gym) & SW(University,Gym). SE(Gallery,Gym) & SE(Museum,Gym) & SE(Stadium,Gym) & SE(Veterinary Clinic,Gym). N(High School,Gym). S(Hospital,Gym). Qualifying objects = {High School}.

Case 1 falsifies the queried relation for {Florist, Gallery, Hospital, Hotel, Museum, Post Office, Restaurant, Stadium, University, Veterinary Clinic}; none is entailed.

Only High School is guaranteed to be north of Gym.

High School is not listed among A-D.

Answer: F

## which: multiple-entailed (first)

ID: `controlled-train-v2-q02088-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
Stadium is to the Southeast of Clinic. Clinic is to the South of Post Office. Clinic is to the Northwest of Bakery. Courthouse is to the Southwest of Clinic. Museum is to the Southeast of Clinic. Clinic is to the Northeast of High School. Stadium is to the Northwest of Zoo. Clinic is to the South of Cafe. University is to the North of Clinic. Clinic is to the Northwest of Zoo. Bank is to the Southwest of Clinic. Clinic is to the Northwest of Hospital.

Which object is north of Clinic?
A. University
B. Hospital
C. Post Office
D. Cafe
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P2: Clinic is to the South of Post Office.
P8: Clinic is to the South of Cafe.

P8 places Cafe north of Clinic.

P2 places Post Office north of Clinic.

There is more than one guaranteed answer.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P2: Clinic is to the South of Post Office.
P8: Clinic is to the South of Cafe.

[P8] N(Cafe,Clinic).

[P2] N(Post Office,Clinic).

There is more than one guaranteed answer.

Answer: E

## which: A-D (L8)

ID: `controlled-train-v2-q03134-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
Gas Station is to the Northwest of Fire Station. Train Station is to the Northwest of University. Bakery is to the Northwest of Cinema. Gas Station is to the Southwest of Clinic. Pharmacy is to the Northwest of University. Veterinary Clinic is to the East of Gas Station. Restaurant is to the Southwest of Cafe. Cinema is to the Northwest of Courthouse. Gas Station is to the Northwest of Bank. Pharmacy is to the Northwest of Train Station. Restaurant is to the West of Hospital. Gas Station is to the West of Hospital. Bakery is to the Southeast of Hotel. Gas Station is to the Southwest of Park. Hotel is to the Southeast of Gallery. Pharmacy is to the Southeast of City Hall. Gas Station is to the Southwest of City Hall. Cafe is to the Northwest of Gallery. Pharmacy is to the Southwest of Clinic. Gas Station is to the Southwest of High School. Clinic is to the Southeast of Park. Police Station is to the Southeast of Courthouse. Police Station is to the Northwest of Gas Station.

Which object is west of Gas Station?
A. University
B. Gallery
C. Restaurant
D. Park
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P3: Bakery is to the Northwest of Cinema.
P7: Restaurant is to the Southwest of Cafe.
P8: Cinema is to the Northwest of Courthouse.
P11: Restaurant is to the West of Hospital.
P12: Gas Station is to the West of Hospital.
P13: Bakery is to the Southeast of Hotel.
P15: Hotel is to the Southeast of Gallery.
P18: Cafe is to the Northwest of Gallery.
P22: Police Station is to the Southeast of Courthouse.
P23: Police Station is to the Northwest of Gas Station.

From west to east: Restaurant, Cafe, Gallery, Hotel, Bakery, Cinema, Courthouse, Police Station and Gas Station (P7, P18, P15, P13, P3, P8, P22, P23).
Restaurant, Hospital and Gas Station lie on the same horizontal line (P11, P12).
Together, these relations place Restaurant west of Gas Station.

Case 1: One arrangement satisfying the premises is as follows. Bakery, Cafe, Cinema, Courthouse, Gallery, Hotel and Police Station are northwest of Gas Station. Bank, Fire Station, Pharmacy, Train Station and University are southeast of Gas Station. City Hall, Clinic, High School and Park are northeast of Gas Station. Hospital and Veterinary Clinic are east of Gas Station. Restaurant is west of Gas Station. Only Restaurant qualifies in this arrangement.

Bakery, Bank, Cafe, Cinema, City Hall, Clinic, Courthouse, Fire Station, Gallery, High School, Hospital, Hotel, Park, Pharmacy, Police Station, Train Station, University and Veterinary Clinic are not west of Gas Station in this arrangement, so none is guaranteed to be.

Only Restaurant is guaranteed to be west of Gas Station.

Restaurant is option C.

Answer: C

### Symbolic

Premises used (labels follow their order in the question):
P3: Bakery is to the Northwest of Cinema.
P7: Restaurant is to the Southwest of Cafe.
P8: Cinema is to the Northwest of Courthouse.
P11: Restaurant is to the West of Hospital.
P12: Gas Station is to the West of Hospital.
P13: Bakery is to the Southeast of Hotel.
P15: Hotel is to the Southeast of Gallery.
P18: Cafe is to the Northwest of Gallery.
P22: Police Station is to the Southeast of Courthouse.
P23: Police Station is to the Northwest of Gas Station.

[P7, P18, P15, P13, P3, P8, P22, P23] x(Restaurant) < x(Cafe) < x(Gallery) < x(Hotel) < x(Bakery) < x(Cinema) < x(Courthouse) < x(Police Station) < x(Gas Station).
[P11, P12] y(Restaurant) = y(Hospital) = y(Gas Station).
Therefore W(Restaurant,Gas Station).

Case 1 (satisfies the premises): NW(Bakery,Gas Station) & NW(Cafe,Gas Station) & NW(Cinema,Gas Station) & NW(Courthouse,Gas Station) & NW(Gallery,Gas Station) & NW(Hotel,Gas Station) & NW(Police Station,Gas Station). SE(Bank,Gas Station) & SE(Fire Station,Gas Station) & SE(Pharmacy,Gas Station) & SE(Train Station,Gas Station) & SE(University,Gas Station). NE(City Hall,Gas Station) & NE(Clinic,Gas Station) & NE(High School,Gas Station) & NE(Park,Gas Station). E(Hospital,Gas Station) & E(Veterinary Clinic,Gas Station). W(Restaurant,Gas Station). Qualifying objects = {Restaurant}.

Case 1 falsifies the queried relation for {Bakery, Bank, Cafe, Cinema, City Hall, Clinic, Courthouse, Fire Station, Gallery, High School, Hospital, Hotel, Park, Pharmacy, Police Station, Train Station, University, Veterinary Clinic}; none is entailed.

Only Restaurant is guaranteed to be west of Gas Station.

Restaurant is option C.

Answer: C

## which: sole with contingent alternative (first)

ID: `controlled-train-v2-q03400-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
Grocery Store is to the Northwest of Bank. Florist is to the Southwest of Grocery Store. Grocery Store is to the Northeast of Bakery. Supermarket is to the Southeast of Grocery Store. Grocery Store is to the Northeast of Bookstore. Grocery Store is to the Northeast of University. Clinic is to the Southwest of Grocery Store. Hospital is to the North of Grocery Store. Park is to the Southwest of Grocery Store. Park is to the Northwest of Bookstore. Veterinary Clinic is to the Southwest of Grocery Store. Courthouse is to the Northeast of University.

Which object is north of Grocery Store?
A. Supermarket
B. Hospital
C. Park
D. University
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P8: Hospital is to the North of Grocery Store.

P8 places Hospital north of Grocery Store.

Case 1: One arrangement satisfying the premises is as follows. Bakery, Bookstore, Clinic, Courthouse, Florist, Park, University and Veterinary Clinic are southwest of Grocery Store. Bank and Supermarket are southeast of Grocery Store. Hospital is north of Grocery Store. Only Hospital qualifies in this arrangement.

Bakery, Bank, Bookstore, Clinic, Courthouse, Florist, Park, Supermarket, University and Veterinary Clinic are not north of Grocery Store in this arrangement, so none is guaranteed to be.

Only Hospital is guaranteed to be north of Grocery Store.

Hospital is option B.

Answer: B

### Symbolic

Premises used (labels follow their order in the question):
P8: Hospital is to the North of Grocery Store.

[P8] N(Hospital,Grocery Store).

Case 1 (satisfies the premises): SW(Bakery,Grocery Store) & SW(Bookstore,Grocery Store) & SW(Clinic,Grocery Store) & SW(Courthouse,Grocery Store) & SW(Florist,Grocery Store) & SW(Park,Grocery Store) & SW(University,Grocery Store) & SW(Veterinary Clinic,Grocery Store). SE(Bank,Grocery Store) & SE(Supermarket,Grocery Store). N(Hospital,Grocery Store). Qualifying objects = {Hospital}.

Case 1 falsifies the queried relation for {Bakery, Bank, Bookstore, Clinic, Courthouse, Florist, Park, Supermarket, University, Veterinary Clinic}; none is entailed.

Only Hospital is guaranteed to be north of Grocery Store.

Hospital is option B.

Answer: B

## which: no-entailed-answer (first)

ID: `controlled-train-v2-q03700-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
Hospital is to the Southwest of Veterinary Clinic. Veterinary Clinic is to the Northeast of Pharmacy. Pharmacy is to the South of Fire Station. Cafe is to the North of Zoo. Pharmacy is to the South of Cafe. Zoo is to the North of Pharmacy.

Which object is south of Pharmacy?
A. Fire Station
B. Zoo
C. Hospital
D. Veterinary Clinic
E. Cannot be determined
F. None of the above

### Natural

Case 1: One arrangement satisfying the premises is as follows. Cafe, Fire Station and Zoo are north of Pharmacy. Hospital is southwest of Pharmacy. Veterinary Clinic is northeast of Pharmacy. No object qualifies in this arrangement.

Cafe, Fire Station, Hospital, Veterinary Clinic and Zoo are not south of Pharmacy in this arrangement, so none is guaranteed to be.

No object is guaranteed to be south of Pharmacy.

Answer: E

### Symbolic

Case 1 (satisfies the premises): N(Cafe,Pharmacy) & N(Fire Station,Pharmacy) & N(Zoo,Pharmacy). SW(Hospital,Pharmacy). NE(Veterinary Clinic,Pharmacy). Qualifying objects = {}.

Case 1 falsifies the queried relation for {Cafe, Fire Station, Hospital, Veterinary Clinic, Zoo}; none is entailed.

No object is guaranteed to be south of Pharmacy.

Answer: E

## which: inconsistent (first)

ID: `controlled-train-v2-q03925-a0-which-m0-unique-entailed`

Consider a map with multiple objects:
Restaurant is to the Northwest of Hotel. Park is to the Northeast of Restaurant. Library is to the Northeast of Restaurant. Fire Station is to the Northeast of Restaurant. Florist is to the Southwest of Restaurant. Restaurant is to the Southwest of Police Station. Gallery is to the Northeast of Restaurant. Florist is to the Northeast of Courthouse. Courthouse is to the Northeast of Restaurant.

Which object is southwest of Restaurant?
A. Florist
B. Fire Station
C. Police Station
D. Gallery
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: Florist is to the Southwest of Restaurant.
P8: Florist is to the Northeast of Courthouse.
P9: Courthouse is to the Northeast of Restaurant.

From west to east: Florist, Restaurant, Courthouse and Florist (P5, P9, P8).
This would put Florist further west than itself, which is impossible.

The premises contradict one another.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P5: Florist is to the Southwest of Restaurant.
P8: Florist is to the Northeast of Courthouse.
P9: Courthouse is to the Northeast of Restaurant.

[P5, P9, P8] x(Florist) < x(Restaurant) < x(Courthouse) < x(Florist).
A strict ordering cannot return to its starting point.

The premises contradict one another.

Answer: E

## count: A-D (first)

ID: `controlled-train-v2-q04000-a0-count-m0-unique-entailed`

Consider a map with multiple objects:
Bookstore is to the South of Gallery. Restaurant is to the South of Gallery. Gallery is to the North of Gas Station. Clinic is to the North of Restaurant. Clinic is to the South of Gallery. Bank is to the South of Gallery.

How many objects are north of Gallery?
A. 3
B. 0
C. 5
D. 2
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Bookstore is to the South of Gallery.
P2: Restaurant is to the South of Gallery.
P3: Gallery is to the North of Gas Station.
P5: Clinic is to the South of Gallery.
P6: Bank is to the South of Gallery.

Bank (P6), Bookstore (P1), Clinic (P5), Gas Station (P3) and Restaurant (P2) lie further south than Gallery, so none can be exactly north of Gallery.

Every other scene object has been accounted for. The premises fix the count at 0.

0 is option B.

Answer: B

### Symbolic

Premises used (labels follow their order in the question):
P1: Bookstore is to the South of Gallery.
P2: Restaurant is to the South of Gallery.
P3: Gallery is to the North of Gas Station.
P5: Clinic is to the South of Gallery.
P6: Bank is to the South of Gallery.

For o in {Bank (P6); Bookstore (P1); Clinic (P5); Gas Station (P3); Restaurant (P2)}: y(o) < y(Gallery) => NOT N(o,Gallery).

All scene memberships are fixed; count=0.

0 is option B.

Answer: B

## count: unique-unlisted (first)

ID: `controlled-train-v2-q04020-a0-count-m1-unique-entailed`

Consider a map with multiple objects:
Zoo is to the North of Gym. Bakery is to the North of Gym. Post Office is to the North of Gym. Gas Station is to the North of Gym. Train Station is to the North of Gas Station. Gym is to the South of Train Station.

How many objects are south of Gym?
A. 1
B. 5
C. 3
D. 2
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Zoo is to the North of Gym.
P2: Bakery is to the North of Gym.
P3: Post Office is to the North of Gym.
P4: Gas Station is to the North of Gym.
P6: Gym is to the South of Train Station.

Bakery (P2), Gas Station (P4), Post Office (P3), Train Station (P6) and Zoo (P1) lie further north than Gym, so none can be exactly south of Gym.

Every other scene object has been accounted for. The premises fix the count at 0.

0 is not listed among A-D.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P1: Zoo is to the North of Gym.
P2: Bakery is to the North of Gym.
P3: Post Office is to the North of Gym.
P4: Gas Station is to the North of Gym.
P6: Gym is to the South of Train Station.

For o in {Bakery (P2); Gas Station (P4); Post Office (P3); Train Station (P6); Zoo (P1)}: y(o) > y(Gym) => NOT S(o,Gym).

All scene memberships are fixed; count=0.

0 is not listed among A-D.

Answer: F

## count: A-D (L8)

ID: `controlled-train-v2-q05134-a0-count-m0-unique-entailed`

Consider a map with multiple objects:
Bookstore is to the Southeast of Gym. Bank is to the Southwest of High School. Clinic is to the Northeast of Bank. Grocery Store is to the Southeast of City Hall. Supermarket is to the Northwest of Florist. Bank is to the Northwest of Pharmacy. Museum is to the Southwest of Gas Station. Park is to the Northeast of Museum. Supermarket is to the Southeast of Grocery Store. Florist is to the Northwest of Fire Station. Museum is to the Northwest of University. Gym is to the Southeast of Fire Station. Pharmacy is to the Southwest of High School. Stadium is to the Southwest of Bank. Cinema is to the Northwest of Pharmacy. Bakery is to the East of Museum. Bookstore is to the Southwest of Zoo. Museum is to the Northwest of Bank. Zoo is to the West of Bakery. City Hall is to the Southeast of Museum. Stadium is to the Southeast of Museum. Veterinary Clinic is to the Northeast of Museum. Museum is to the Northwest of Cinema.

How many objects are west of Museum?
A. 15
B. 7
C. 0
D. 11
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Bookstore is to the Southeast of Gym.
P2: Bank is to the Southwest of High School.
P3: Clinic is to the Northeast of Bank.
P4: Grocery Store is to the Southeast of City Hall.
P5: Supermarket is to the Northwest of Florist.
P6: Bank is to the Northwest of Pharmacy.
P7: Museum is to the Southwest of Gas Station.
P8: Park is to the Northeast of Museum.
P9: Supermarket is to the Southeast of Grocery Store.
P10: Florist is to the Northwest of Fire Station.
P11: Museum is to the Northwest of University.
P12: Gym is to the Southeast of Fire Station.
P16: Bakery is to the East of Museum.
P17: Bookstore is to the Southwest of Zoo.
P18: Museum is to the Northwest of Bank.
P19: Zoo is to the West of Bakery.
P20: City Hall is to the Southeast of Museum.
P21: Stadium is to the Southeast of Museum.
P22: Veterinary Clinic is to the Northeast of Museum.
P23: Museum is to the Northwest of Cinema.

Chain 1: From west to east: Museum, City Hall, Grocery Store, Supermarket, Florist, Fire Station, Gym, Bookstore and Zoo (P20, P4, P9, P5, P10, P12, P1, P17).

Chain 2: Bookstore lies further south than Zoo; Zoo and Bakery lie on the same horizontal line; Bakery and Museum lie on the same horizontal line (P17, P19, P16).

Chain 3: From west to east: Museum, Bank and Clinic (P18, P3).

Chain 4: Reuse Chain 3 from Museum to Bank; then From west to east: Bank and High School (P2).

Chain 5: Reuse Chain 3 from Museum to Bank; then From west to east: Bank and Pharmacy (P6).

Bakery (P16), Bank (Chain 3), Cinema (P23), City Hall (Chain 1), Clinic (Chain 3), Fire Station (Chain 1), Florist (Chain 1), Gas Station (P7), Grocery Store (Chain 1), Gym (Chain 1), High School (Chain 4), Park (P8), Pharmacy (Chain 5), Stadium (P21), Supermarket (Chain 1), University (P11), Veterinary Clinic (P22) and Zoo (Chain 1) lie further east than Museum, so none can be exactly west of Museum.

Bookstore (Chain 2) lies further south than Museum, so it cannot be exactly west of Museum.

Every other scene object has been accounted for. The premises fix the count at 0.

0 is option C.

Answer: C

### Symbolic

Premises used (labels follow their order in the question):
P1: Bookstore is to the Southeast of Gym.
P2: Bank is to the Southwest of High School.
P3: Clinic is to the Northeast of Bank.
P4: Grocery Store is to the Southeast of City Hall.
P5: Supermarket is to the Northwest of Florist.
P6: Bank is to the Northwest of Pharmacy.
P7: Museum is to the Southwest of Gas Station.
P8: Park is to the Northeast of Museum.
P9: Supermarket is to the Southeast of Grocery Store.
P10: Florist is to the Northwest of Fire Station.
P11: Museum is to the Northwest of University.
P12: Gym is to the Southeast of Fire Station.
P16: Bakery is to the East of Museum.
P17: Bookstore is to the Southwest of Zoo.
P18: Museum is to the Northwest of Bank.
P19: Zoo is to the West of Bakery.
P20: City Hall is to the Southeast of Museum.
P21: Stadium is to the Southeast of Museum.
P22: Veterinary Clinic is to the Northeast of Museum.
P23: Museum is to the Northwest of Cinema.

Chain 1: [P20, P4, P9, P5, P10, P12, P1, P17] x(Museum) < x(City Hall) < x(Grocery Store) < x(Supermarket) < x(Florist) < x(Fire Station) < x(Gym) < x(Bookstore) < x(Zoo).

Chain 2: [P17, P19, P16] y(Bookstore) < y(Zoo) = y(Bakery) = y(Museum).

Chain 3: [P18, P3] x(Museum) < x(Bank) < x(Clinic).

Chain 4: Reuse Chain 3 from Museum to Bank; then [P2] x(Bank) < x(High School).

Chain 5: Reuse Chain 3 from Museum to Bank; then [P6] x(Bank) < x(Pharmacy).

For o in {Bakery (P16); Bank (Chain 3); Cinema (P23); City Hall (Chain 1); Clinic (Chain 3); Fire Station (Chain 1); Florist (Chain 1); Gas Station (P7); Grocery Store (Chain 1); Gym (Chain 1); High School (Chain 4); Park (P8); Pharmacy (Chain 5); Stadium (P21); Supermarket (Chain 1); University (P11); Veterinary Clinic (P22); Zoo (Chain 1)}: x(o) > x(Museum) => NOT W(o,Museum).

For o in {Bookstore (Chain 2)}: y(o) < y(Museum) => NOT W(o,Museum).

All scene memberships are fixed; count=0.

0 is option C.

Answer: C

## count: no-entailed-answer (first)

ID: `controlled-train-v2-q05400-a0-count-m0-unique-entailed`

Consider a map with multiple objects:
Bookstore is to the Northeast of City Hall. City Hall is to the Southwest of Museum. Hotel is to the South of Bookstore. Bookstore is to the North of Grocery Store. Bank is to the South of Bookstore. Bank is to the North of Hotel.

How many objects are north of Bookstore?
A. 3
B. 0
C. 4
D. 2
E. Cannot be determined
F. None of the above

### Natural

Case 1: One arrangement satisfying the premises is as follows. Bank, Grocery Store and Hotel are south of Bookstore. City Hall is southwest of Bookstore. Museum is north of Bookstore. Only Museum qualifies in this arrangement. This gives a count of 1.

Case 2: One arrangement satisfying the premises is as follows. Bank, Grocery Store and Hotel are south of Bookstore. City Hall is southwest of Bookstore. Museum is northeast of Bookstore. No object qualifies in this arrangement. This gives a count of 0.

The same premises allow different outcomes, so no exact count is guaranteed.

Answer: E

### Symbolic

Case 1 (satisfies the premises): S(Bank,Bookstore) & S(Grocery Store,Bookstore) & S(Hotel,Bookstore). SW(City Hall,Bookstore). N(Museum,Bookstore). Qualifying objects = {Museum}. count=1.

Case 2 (satisfies the premises): S(Bank,Bookstore) & S(Grocery Store,Bookstore) & S(Hotel,Bookstore). SW(City Hall,Bookstore). NE(Museum,Bookstore). Qualifying objects = {}. count=0.

The same premises allow different outcomes, so no exact count is guaranteed.

Answer: E

## count: inconsistent (first)

ID: `controlled-train-v2-q05850-a0-count-m0-unique-entailed`

Consider a map with multiple objects:
Museum is to the East of Pharmacy. Museum is to the West of Pharmacy. Police Station is to the West of Pharmacy. Pharmacy is to the East of Train Station. Library is to the West of Pharmacy. Pharmacy is to the East of Post Office.

How many objects are east of Pharmacy?
A. 1
B. 3
C. 5
D. 4
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Museum is to the East of Pharmacy.
P2: Museum is to the West of Pharmacy.

From west to east: Pharmacy, Museum and Pharmacy (P1, P2).
This would put Pharmacy further west than itself, which is impossible.

The premises contradict one another.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P1: Museum is to the East of Pharmacy.
P2: Museum is to the West of Pharmacy.

[P1, P2] x(Pharmacy) < x(Museum) < x(Pharmacy).
A strict ordering cannot return to its starting point.

The premises contradict one another.

Answer: E
