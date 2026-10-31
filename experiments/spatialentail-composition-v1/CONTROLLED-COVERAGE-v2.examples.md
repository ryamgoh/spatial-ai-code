# Controlled Compact Examples

Verified development examples; existing renderers unchanged.

## count: inconsistent

Consider a map with multiple objects:
Florist is to the South of Veterinary Clinic. Veterinary Clinic is to the North of Park. Veterinary Clinic is to the North of University. Veterinary Clinic is to the North of Fire Station. Veterinary Clinic is to the South of Park. Veterinary Clinic is to the North of Gas Station.

How many objects are north of Veterinary Clinic?
A. 5
B. 4
C. 1
D. 2
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P2: Veterinary Clinic is to the North of Park.
P5: Veterinary Clinic is to the South of Park.

From P2, Park's y-coordinate is less than Veterinary Clinic's y-coordinate.
From P5, Veterinary Clinic's y-coordinate is less than Park's y-coordinate.
Composing this path gives Park's y-coordinate less than Park's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: P2, P5. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P2: Veterinary Clinic is to the North of Park.
P5: Veterinary Clinic is to the South of Park.

P2: y(Park) < y(Veterinary Clinic)
P5: y(Veterinary Clinic) < y(Park)
Transitivity: y(Park) < y(Park)
A strict order cannot be less than itself: contradiction.
Conflicting premises: P2, P5. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

## count: listed

Consider a map with multiple objects:
City Hall is to the North of Cinema. Cafe is to the South of City Hall. City Hall is to the North of Zoo. City Hall is to the North of Gallery. Gallery is to the South of Cinema. Supermarket is to the South of City Hall.

How many objects are north of City Hall?
A. 1
B. 2
C. 0
D. 4
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: City Hall is to the North of Cinema.
P2: Cafe is to the South of City Hall.
P3: City Hall is to the North of Zoo.
P4: City Hall is to the North of Gallery.
P6: Supermarket is to the South of City Hall.

City Hall has a greater y-coordinate than Cafe (premise sources: P2).
Cafe never qualifies for the queried relation.
City Hall has a greater y-coordinate than Cinema (premise sources: P1).
Cinema never qualifies for the queried relation.
City Hall has a greater y-coordinate than Gallery (premise sources: P4).
Gallery never qualifies for the queried relation.
City Hall has a greater y-coordinate than Supermarket (premise sources: P6).
Supermarket never qualifies for the queried relation.
City Hall has a greater y-coordinate than Zoo (premise sources: P3).
Zoo never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Cafe, Cinema, Gallery, Supermarket and Zoo are south of City Hall. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

### Symbolic

Premises used (labels follow their order in the question):
P1: City Hall is to the North of Cinema.
P2: Cafe is to the South of City Hall.
P3: City Hall is to the North of Zoo.
P4: City Hall is to the North of Gallery.
P6: Supermarket is to the South of City Hall.

[P2] y(Cafe) < y(City Hall).
NOT N(Cafe,City Hall) in every valid arrangement.
[P1] y(Cinema) < y(City Hall).
NOT N(Cinema,City Hall) in every valid arrangement.
[P4] y(Gallery) < y(City Hall).
NOT N(Gallery,City Hall) in every valid arrangement.
[P6] y(Supermarket) < y(City Hall).
NOT N(Supermarket,City Hall) in every valid arrangement.
[P3] y(Zoo) < y(City Hall).
NOT N(Zoo,City Hall) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

Case 1 (counterexample satisfying all premises): S(Cafe,City Hall) & S(Cinema,City Hall) & S(Gallery,City Hall) & S(Supermarket,City Hall) & S(Zoo,City Hall). Qualifying objects = {}. count=0.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

## count: no-entailed

Consider a map with multiple objects:
Pharmacy is to the North of Clinic. Police Station is to the Southwest of Museum. Police Station is to the Southwest of Restaurant. University is to the South of Restaurant. Clinic is to the South of Restaurant. Restaurant is to the North of Pharmacy.

How many objects are north of Restaurant?
A. 5
B. 0
C. 2
D. 1
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Clinic, Pharmacy and University are south of Restaurant. Museum is north of Restaurant. Police Station is southwest of Restaurant. Only Museum qualifies in this arrangement. This gives a count of 1.

Case 2: A counterexample arrangement satisfying all premises is the following. Clinic, Pharmacy and University are south of Restaurant. Museum and Police Station are southwest of Restaurant. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 2, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 2, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 2, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 2, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 2, so it is not entailed.

No domain answer is entailed.

Answer: E

### Symbolic

Case 1 (counterexample satisfying all premises): S(Clinic,Restaurant) & S(Pharmacy,Restaurant) & S(University,Restaurant). N(Museum,Restaurant). SW(Police Station,Restaurant). Qualifying objects = {Museum}. count=1.

Case 2 (counterexample satisfying all premises): S(Clinic,Restaurant) & S(Pharmacy,Restaurant) & S(University,Restaurant). SW(Museum,Restaurant) & SW(Police Station,Restaurant). Qualifying objects = {}. count=0.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 2, so it is not entailed.

The claim that count=2 is false in Case 2, so it is not entailed.

The claim that count=3 is false in Case 2, so it is not entailed.

The claim that count=4 is false in Case 2, so it is not entailed.

The claim that count=5 is false in Case 2, so it is not entailed.

No domain answer is entailed.

Answer: E

## count: unlisted

Consider a map with multiple objects:
Supermarket is to the North of Pharmacy. Bookstore is to the South of Supermarket. Supermarket is to the North of Veterinary Clinic. Zoo is to the South of Supermarket. Post Office is to the South of Supermarket. Veterinary Clinic is to the South of Zoo.

How many objects are north of Supermarket?
A. 4
B. 2
C. 1
D. 3
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Supermarket is to the North of Pharmacy.
P2: Bookstore is to the South of Supermarket.
P3: Supermarket is to the North of Veterinary Clinic.
P4: Zoo is to the South of Supermarket.
P5: Post Office is to the South of Supermarket.

Supermarket has a greater y-coordinate than Bookstore (premise sources: P2).
Bookstore never qualifies for the queried relation.
Supermarket has a greater y-coordinate than Pharmacy (premise sources: P1).
Pharmacy never qualifies for the queried relation.
Supermarket has a greater y-coordinate than Post Office (premise sources: P5).
Post Office never qualifies for the queried relation.
Supermarket has a greater y-coordinate than Veterinary Clinic (premise sources: P3).
Veterinary Clinic never qualifies for the queried relation.
Supermarket has a greater y-coordinate than Zoo (premise sources: P4).
Zoo never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P1: Supermarket is to the North of Pharmacy.
P2: Bookstore is to the South of Supermarket.
P3: Supermarket is to the North of Veterinary Clinic.
P4: Zoo is to the South of Supermarket.
P5: Post Office is to the South of Supermarket.

[P2] y(Bookstore) < y(Supermarket).
NOT N(Bookstore,Supermarket) in every valid arrangement.
[P1] y(Pharmacy) < y(Supermarket).
NOT N(Pharmacy,Supermarket) in every valid arrangement.
[P5] y(Post Office) < y(Supermarket).
NOT N(Post Office,Supermarket) in every valid arrangement.
[P3] y(Veterinary Clinic) < y(Supermarket).
NOT N(Veterinary Clinic,Supermarket) in every valid arrangement.
[P4] y(Zoo) < y(Supermarket).
NOT N(Zoo,Supermarket) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

## direction: inconsistent

Consider a map with multiple objects:
Hotel is to the South of Hospital. Hotel is to the North of Hospital. Bookstore is to the South of Hospital. Gym is to the South of Hospital. Library is to the South of Hospital. Hospital is to the North of Zoo.

In which direction is Hotel relative to Hospital?
A. Northeast
B. East
C. Southwest
D. Southeast
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P1: Hotel is to the South of Hospital.
P2: Hotel is to the North of Hospital.

From P1, Hotel's y-coordinate is less than Hospital's y-coordinate.
From P2, Hospital's y-coordinate is less than Hotel's y-coordinate.
Composing this path gives Hotel's y-coordinate less than Hotel's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: P1, P2. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P1: Hotel is to the South of Hospital.
P2: Hotel is to the North of Hospital.

P1: y(Hotel) < y(Hospital)
P2: y(Hospital) < y(Hotel)
Transitivity: y(Hotel) < y(Hotel)
A strict order cannot be less than itself: contradiction.
Conflicting premises: P1, P2. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

## direction: listed

Consider a map with multiple objects:
Cafe is to the South of Bookstore. Cafe is to the North of Gas Station. Bookstore is to the North of Courthouse. Bookstore is to the North of Zoo. Gas Station is to the South of Bookstore. Hospital is to the North of Bookstore.

In which direction is Hospital relative to Bookstore?
A. Southwest
B. Northwest
C. West
D. North
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P6: Hospital is to the North of Bookstore.

Bookstore and Hospital share the same x-coordinate (premise sources: P6).
Hospital has a greater y-coordinate than Bookstore (premise sources: P6).
Therefore, Hospital is north of Bookstore in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bookstore and Hospital share the same horizontal position. From bottom to top, the order is Bookstore and Hospital. So Hospital is north of Bookstore.

The claim that Hospital is northeast of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is east of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is southeast of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is south of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is southwest of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is west of Bookstore is false in Case 1, so it is not entailed.

The claim that Hospital is northwest of Bookstore is false in Case 1, so it is not entailed.

The complete entailed-answer set is {North}.

Answer: D

### Symbolic

Premises used (labels follow their order in the question):
P6: Hospital is to the North of Bookstore.

[P6] x(Bookstore) = x(Hospital).
[P6] y(Bookstore) < y(Hospital).
N(Hospital,Bookstore) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Bookstore) = x(Hospital). y(Bookstore) < y(Hospital). => N(Hospital,Bookstore).

The claim that NE(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that E(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that SE(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that S(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that SW(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that W(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The claim that NW(Hospital,Bookstore) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {North}.

Answer: D

## direction: no-entailed

Consider a map with multiple objects:
Pharmacy is to the North of Library. Grocery Store is to the South of Pharmacy. Grocery Store is to the North of Library. Pharmacy is to the Northeast of High School. Pharmacy is to the North of Train Station. High School is to the Southwest of Florist.

In which direction is Florist relative to Pharmacy?
A. East
B. Southwest
C. West
D. South
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. From left to right, the order is High School, Florist and Pharmacy. From bottom to top, the order is High School, Florist and Pharmacy. So Florist is southwest of Pharmacy.

Case 2: A counterexample arrangement satisfying all premises is the following. High School lies to the left of Florist and Pharmacy. Florist and Pharmacy share the same horizontal position. From bottom to top, the order is High School, Pharmacy and Florist. So Florist is north of Pharmacy.

The claim that Florist is north of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is northeast of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is east of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is southeast of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is south of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is southwest of Pharmacy is false in Case 2, so it is not entailed.

The claim that Florist is west of Pharmacy is false in Case 1, so it is not entailed.

The claim that Florist is northwest of Pharmacy is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

### Symbolic

Case 1 (counterexample satisfying all premises): x(High School) < x(Florist) < x(Pharmacy). y(High School) < y(Florist) < y(Pharmacy). => SW(Florist,Pharmacy).

Case 2 (counterexample satisfying all premises): x(High School) < x(Florist) = x(Pharmacy). y(High School) < y(Pharmacy) < y(Florist). => N(Florist,Pharmacy).

The claim that N(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that NE(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that E(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that SE(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that S(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that SW(Florist,Pharmacy) is false in Case 2, so it is not entailed.

The claim that W(Florist,Pharmacy) is false in Case 1, so it is not entailed.

The claim that NW(Florist,Pharmacy) is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

## direction: unlisted

Consider a map with multiple objects:
Cinema is to the North of Courthouse. Courthouse is to the South of Florist. Florist is to the North of Cinema. Veterinary Clinic is to the South of Florist. Florist is to the South of Museum. Florist is to the North of Zoo.

In which direction is Museum relative to Florist?
A. Southwest
B. Northwest
C. Northeast
D. West
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: Florist is to the South of Museum.

Florist and Museum share the same x-coordinate (premise sources: P5).
Museum has a greater y-coordinate than Florist (premise sources: P5).
Therefore, Museum is north of Florist in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P5: Florist is to the South of Museum.

[P5] x(Florist) = x(Museum).
[P5] y(Florist) < y(Museum).
N(Museum,Florist) in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

## which: inconsistent

Consider a map with multiple objects:
Zoo is to the South of Hospital. Police Station is to the South of Hospital. Hospital is to the North of Cafe. City Hall is to the South of Hospital. Hospital is to the South of Supermarket. Hospital is to the North of Supermarket.

Which object is north of Hospital?
A. Zoo
B. Cafe
C. Supermarket
D. City Hall
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: Hospital is to the South of Supermarket.
P6: Hospital is to the North of Supermarket.

From P5, Hospital's y-coordinate is less than Supermarket's y-coordinate.
From P6, Supermarket's y-coordinate is less than Hospital's y-coordinate.
Composing this path gives Hospital's y-coordinate less than Hospital's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: P5, P6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P5: Hospital is to the South of Supermarket.
P6: Hospital is to the North of Supermarket.

P5: y(Hospital) < y(Supermarket)
P6: y(Supermarket) < y(Hospital)
Transitivity: y(Hospital) < y(Hospital)
A strict order cannot be less than itself: contradiction.
Conflicting premises: P5, P6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

## which: listed

Consider a map with multiple objects:
High School is to the South of Courthouse. Courthouse is to the North of Theater. Museum is to the South of Courthouse. High School is to the South of Veterinary Clinic. Police Station is to the North of Courthouse. Courthouse is to the North of Veterinary Clinic.

Which object is north of Courthouse?
A. Veterinary Clinic
B. Theater
C. High School
D. Police Station
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: Police Station is to the North of Courthouse.

Courthouse and Police Station share the same x-coordinate (premise sources: P5).
Police Station has a greater y-coordinate than Courthouse (premise sources: P5).
Therefore, Police Station is north of Courthouse in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. High School, Museum, Theater and Veterinary Clinic are south of Courthouse. Police Station is north of Courthouse. Only Police Station qualifies in this arrangement.

The claim that High School is north of Courthouse is false in Case 1, so it is not entailed.

The claim that Museum is north of Courthouse is false in Case 1, so it is not entailed.

The claim that Theater is north of Courthouse is false in Case 1, so it is not entailed.

The claim that Veterinary Clinic is north of Courthouse is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Police Station}.

Answer: D

### Symbolic

Premises used (labels follow their order in the question):
P5: Police Station is to the North of Courthouse.

[P5] x(Courthouse) = x(Police Station).
[P5] y(Courthouse) < y(Police Station).
N(Police Station,Courthouse) in every valid arrangement.

Case 1 (counterexample satisfying all premises): S(High School,Courthouse) & S(Museum,Courthouse) & S(Theater,Courthouse) & S(Veterinary Clinic,Courthouse). N(Police Station,Courthouse). Qualifying objects = {Police Station}.

The claim that N(High School,Courthouse) is false in Case 1, so it is not entailed.

The claim that N(Museum,Courthouse) is false in Case 1, so it is not entailed.

The claim that N(Theater,Courthouse) is false in Case 1, so it is not entailed.

The claim that N(Veterinary Clinic,Courthouse) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Police Station}.

Answer: D

## which: multiple

Consider a map with multiple objects:
High School is to the South of Gallery. Courthouse is to the South of Gas Station. Gas Station is to the North of Gallery. Bakery is to the North of Gallery. Supermarket is to the South of Gallery. Courthouse is to the North of Gallery.

Which object is north of Gallery?
A. Courthouse
B. Gas Station
C. Bakery
D. Supermarket
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P4: Bakery is to the North of Gallery.
P6: Courthouse is to the North of Gallery.

Gallery and Bakery share the same x-coordinate (premise sources: P4).
Bakery has a greater y-coordinate than Gallery (premise sources: P4).
Therefore, Bakery is north of Gallery in every valid arrangement.

Gallery and Courthouse share the same x-coordinate (premise sources: P6).
Courthouse has a greater y-coordinate than Gallery (premise sources: P6).
Therefore, Courthouse is north of Gallery in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

Premises used (labels follow their order in the question):
P4: Bakery is to the North of Gallery.
P6: Courthouse is to the North of Gallery.

[P4] x(Gallery) = x(Bakery).
[P4] y(Gallery) < y(Bakery).
N(Bakery,Gallery) in every valid arrangement.

[P6] x(Gallery) = x(Courthouse).
[P6] y(Gallery) < y(Courthouse).
N(Courthouse,Gallery) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## which: no-entailed

Consider a map with multiple objects:
Cinema is to the South of Supermarket. Bookstore is to the South of Supermarket. Supermarket is to the North of Restaurant. Supermarket is to the Northeast of City Hall. Stadium is to the Northeast of City Hall. Cinema is to the North of Bookstore.

Which object is north of Supermarket?
A. Restaurant
B. Bookstore
C. City Hall
D. Cinema
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Bookstore, Cinema and Restaurant are south of Supermarket. City Hall is southwest of Supermarket. Stadium is southeast of Supermarket. No eligible object qualifies in this arrangement.

The claim that Bookstore is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Cinema is north of Supermarket is false in Case 1, so it is not entailed.

The claim that City Hall is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Restaurant is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Stadium is north of Supermarket is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

### Symbolic

Case 1 (counterexample satisfying all premises): S(Bookstore,Supermarket) & S(Cinema,Supermarket) & S(Restaurant,Supermarket). SW(City Hall,Supermarket). SE(Stadium,Supermarket). Qualifying objects = {}.

The claim that N(Bookstore,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Cinema,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(City Hall,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Restaurant,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Stadium,Supermarket) is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

## which: unlisted

Consider a map with multiple objects:
Veterinary Clinic is to the North of Museum. City Hall is to the North of Library. Restaurant is to the South of City Hall. Veterinary Clinic is to the South of City Hall. City Hall is to the South of Grocery Store. Museum is to the South of City Hall.

Which object is north of City Hall?
A. Library
B. Museum
C. Restaurant
D. Veterinary Clinic
E. Cannot be determined
F. None of the above

### Natural

Premises used (labels follow their order in the question):
P5: City Hall is to the South of Grocery Store.

City Hall and Grocery Store share the same x-coordinate (premise sources: P5).
Grocery Store has a greater y-coordinate than City Hall (premise sources: P5).
Therefore, Grocery Store is north of City Hall in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Grocery Store is north of City Hall. Library, Museum, Restaurant and Veterinary Clinic are south of City Hall. Only Grocery Store qualifies in this arrangement.

The claim that Library is north of City Hall is false in Case 1, so it is not entailed.

The claim that Museum is north of City Hall is false in Case 1, so it is not entailed.

The claim that Restaurant is north of City Hall is false in Case 1, so it is not entailed.

The claim that Veterinary Clinic is north of City Hall is false in Case 1, so it is not entailed.

The sole entailed answer is unlisted.

Answer: F

### Symbolic

Premises used (labels follow their order in the question):
P5: City Hall is to the South of Grocery Store.

[P5] x(City Hall) = x(Grocery Store).
[P5] y(City Hall) < y(Grocery Store).
N(Grocery Store,City Hall) in every valid arrangement.

Case 1 (counterexample satisfying all premises): N(Grocery Store,City Hall). S(Library,City Hall) & S(Museum,City Hall) & S(Restaurant,City Hall) & S(Veterinary Clinic,City Hall). Qualifying objects = {Grocery Store}.

The claim that N(Library,City Hall) is false in Case 1, so it is not entailed.

The claim that N(Museum,City Hall) is false in Case 1, so it is not entailed.

The claim that N(Restaurant,City Hall) is false in Case 1, so it is not entailed.

The claim that N(Veterinary Clinic,City Hall) is false in Case 1, so it is not entailed.

The sole entailed answer is unlisted.

Answer: F
