# Paired Reference Review Samples

Deterministic first occurrence per review category; not a random quality estimate.

## production-b2758-s5-count-m0-unique-entailed

Categories: query-count, contract-unique-entailed, depth-9, escape-None.
Split: extrapolation-test; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Hotel, Park, Gas Station, Post Office, Fire Station, Restaurant, Stadium, Supermarket, Hospital, Clinic, Train Station, Gym, High School, Bakery, Cinema, Zoo, Museum, Police Station, Courthouse, Cafe, Bank, Gallery, Theater, Florist, Bookstore, Grocery Store, City Hall, Veterinary Clinic, Library, Pharmacy
Query: count Southeast Gym
Eligible: Bakery, Hotel, Cinema, Courthouse, Restaurant, Clinic, Museum, Cafe
Consider a map with multiple objects:
Gym is to the Southeast of Post Office.
Gym is to the Northwest of Museum.
Cafe is to the Northwest of Gym.
Bakery is to the North of Police Station.
Restaurant is to the Northwest of Bookstore.
Train Station is to the Southeast of Bank.
Theater is to the Southeast of City Hall.
Bookstore is to the Southeast of Courthouse.
Fire Station is to the Southeast of Zoo.
Zoo is to the Southeast of Bakery.
Stadium is to the Northwest of Bank.
Veterinary Clinic is to the Northwest of Cinema.
Gallery is to the Northwest of Stadium.
Post Office is to the Southeast of Clinic.
Gym is to the Northwest of Hotel.
High School is to the Southeast of Grocery Store.
Park is to the Southeast of Library.
Restaurant is to the Southeast of Clinic.
Train Station is to the Northwest of Veterinary Clinic.
Pharmacy is to the Northwest of Gallery.
Hospital is to the Northwest of Florist.
Gym is to the Northwest of City Hall.
Gas Station is to the East of Train Station.
Florist is to the Northwest of Supermarket.
Pharmacy is to the Southeast of Theater.
Hospital is to the Southeast of Bookstore.
Fire Station is to the Northwest of Post Office.
Zoo is to the Southeast of Supermarket.
Question: What exact count of eligible objects southeast of Gym is guaranteed?
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Bakery, Zoo, Fire Station, Post Office and Gym (premise sources: 10, 9, 27 and 1). So Gym has a greater x-coordinate than Bakery.
Bakery never qualifies for the queried relation.
Hotel has a greater x-coordinate than Gym (premise sources: 15).
Gym has a greater y-coordinate than Hotel (premise sources: 15).
Hotel always qualifies for the queried relation.
The x-coordinates increase strictly along Gym, City Hall, Theater, Pharmacy, Gallery, Stadium, Bank, Train Station, Veterinary Clinic and Cinema (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Cinema has a greater x-coordinate than Gym.
The y-coordinates increase strictly along Cinema, Veterinary Clinic, Train Station, Bank, Stadium, Gallery, Pharmacy, Theater, City Hall and Gym (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Gym has a greater y-coordinate than Cinema.
Cinema always qualifies for the queried relation.
The x-coordinates increase strictly along Courthouse, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Courthouse.
Courthouse never qualifies for the queried relation.
The x-coordinates increase strictly along Restaurant, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Restaurant.
Restaurant never qualifies for the queried relation.
The x-coordinates increase strictly along Clinic, Post Office and Gym (premise sources: 14 and 1). So Gym has a greater x-coordinate than Clinic.
Clinic never qualifies for the queried relation.
Museum has a greater x-coordinate than Gym (premise sources: 2).
Gym has a greater y-coordinate than Museum (premise sources: 2).
Museum always qualifies for the queried relation.
Gym has a greater x-coordinate than Cafe (premise sources: 3).
Cafe never qualifies for the queried relation.
The qualifying candidates are fixed: Hotel, Cinema and Museum. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bakery, Courthouse, Restaurant, Clinic and Cafe are northwest of Gym. Hotel, Cinema and Museum are southeast of Gym. Only Cinema, Hotel and Museum qualify in this arrangement. This gives a count of 3.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Bakery) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Bakery) < x(Gym).
NOT SE(Bakery,Gym) in every valid arrangement.
[p15] x(Gym) < x(Hotel).
[p15] y(Hotel) < y(Gym).
SE(Hotel,Gym) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Gym) < x(City Hall) < x(Theater) < x(Pharmacy) < x(Gallery) < x(Stadium) < x(Bank) < x(Train Station) < x(Veterinary Clinic) < x(Cinema) => x(Gym) < x(Cinema).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Cinema) < y(Veterinary Clinic) < y(Train Station) < y(Bank) < y(Stadium) < y(Gallery) < y(Pharmacy) < y(Theater) < y(City Hall) < y(Gym) => y(Cinema) < y(Gym).
SE(Cinema,Gym) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Courthouse) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Courthouse) < x(Gym).
NOT SE(Courthouse,Gym) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Restaurant) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Restaurant) < x(Gym).
NOT SE(Restaurant,Gym) in every valid arrangement.
[p14, p1] x(Clinic) < x(Post Office) < x(Gym) => x(Clinic) < x(Gym).
NOT SE(Clinic,Gym) in every valid arrangement.
[p2] x(Gym) < x(Museum).
[p2] y(Museum) < y(Gym).
SE(Museum,Gym) in every valid arrangement.
[p3] x(Cafe) < x(Gym).
NOT SE(Cafe,Gym) in every valid arrangement.
Fixed qualifying set={Hotel, Cinema, Museum}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Bakery,Gym) & NW(Courthouse,Gym) & NW(Restaurant,Gym) & NW(Clinic,Gym) & NW(Cafe,Gym). SE(Hotel,Gym) & SE(Cinema,Gym) & SE(Museum,Gym). Qualifying objects = {Cinema, Hotel, Museum}. count=3.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The claim that count=8 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

## production-b2758-s5-count-m0-listed-entailed

Categories: contract-listed-entailed.
Split: extrapolation-test; contract: listed-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: listed-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all displayed A-D values that are entailed. Exclude every displayed value that is not entailed, including claims that could hold but are not guaranteed. Select F if no displayed value is entailed.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Hotel, Park, Gas Station, Post Office, Fire Station, Restaurant, Stadium, Supermarket, Hospital, Clinic, Train Station, Gym, High School, Bakery, Cinema, Zoo, Museum, Police Station, Courthouse, Cafe, Bank, Gallery, Theater, Florist, Bookstore, Grocery Store, City Hall, Veterinary Clinic, Library, Pharmacy
Query: count Southeast Gym
Eligible: Bakery, Hotel, Cinema, Courthouse, Restaurant, Clinic, Museum, Cafe
Consider a map with multiple objects:
Gym is to the Southeast of Post Office.
Gym is to the Northwest of Museum.
Cafe is to the Northwest of Gym.
Bakery is to the North of Police Station.
Restaurant is to the Northwest of Bookstore.
Train Station is to the Southeast of Bank.
Theater is to the Southeast of City Hall.
Bookstore is to the Southeast of Courthouse.
Fire Station is to the Southeast of Zoo.
Zoo is to the Southeast of Bakery.
Stadium is to the Northwest of Bank.
Veterinary Clinic is to the Northwest of Cinema.
Gallery is to the Northwest of Stadium.
Post Office is to the Southeast of Clinic.
Gym is to the Northwest of Hotel.
High School is to the Southeast of Grocery Store.
Park is to the Southeast of Library.
Restaurant is to the Southeast of Clinic.
Train Station is to the Northwest of Veterinary Clinic.
Pharmacy is to the Northwest of Gallery.
Hospital is to the Northwest of Florist.
Gym is to the Northwest of City Hall.
Gas Station is to the East of Train Station.
Florist is to the Northwest of Supermarket.
Pharmacy is to the Southeast of Theater.
Hospital is to the Southeast of Bookstore.
Fire Station is to the Northwest of Post Office.
Zoo is to the Southeast of Supermarket.
Question: Which listed exact counts of eligible objects southeast of Gym are entailed? Select all that apply.
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Bakery, Zoo, Fire Station, Post Office and Gym (premise sources: 10, 9, 27 and 1). So Gym has a greater x-coordinate than Bakery.
Bakery never qualifies for the queried relation.
Hotel has a greater x-coordinate than Gym (premise sources: 15).
Gym has a greater y-coordinate than Hotel (premise sources: 15).
Hotel always qualifies for the queried relation.
The x-coordinates increase strictly along Gym, City Hall, Theater, Pharmacy, Gallery, Stadium, Bank, Train Station, Veterinary Clinic and Cinema (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Cinema has a greater x-coordinate than Gym.
The y-coordinates increase strictly along Cinema, Veterinary Clinic, Train Station, Bank, Stadium, Gallery, Pharmacy, Theater, City Hall and Gym (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Gym has a greater y-coordinate than Cinema.
Cinema always qualifies for the queried relation.
The x-coordinates increase strictly along Courthouse, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Courthouse.
Courthouse never qualifies for the queried relation.
The x-coordinates increase strictly along Restaurant, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Restaurant.
Restaurant never qualifies for the queried relation.
The x-coordinates increase strictly along Clinic, Post Office and Gym (premise sources: 14 and 1). So Gym has a greater x-coordinate than Clinic.
Clinic never qualifies for the queried relation.
Museum has a greater x-coordinate than Gym (premise sources: 2).
Gym has a greater y-coordinate than Museum (premise sources: 2).
Museum always qualifies for the queried relation.
Gym has a greater x-coordinate than Cafe (premise sources: 3).
Cafe never qualifies for the queried relation.
The qualifying candidates are fixed: Hotel, Cinema and Museum. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bakery, Courthouse, Restaurant, Clinic and Cafe are northwest of Gym. Hotel, Cinema and Museum are southeast of Gym. Only Cinema, Hotel and Museum qualify in this arrangement. This gives a count of 3.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The displayed entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Bakery) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Bakery) < x(Gym).
NOT SE(Bakery,Gym) in every valid arrangement.
[p15] x(Gym) < x(Hotel).
[p15] y(Hotel) < y(Gym).
SE(Hotel,Gym) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Gym) < x(City Hall) < x(Theater) < x(Pharmacy) < x(Gallery) < x(Stadium) < x(Bank) < x(Train Station) < x(Veterinary Clinic) < x(Cinema) => x(Gym) < x(Cinema).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Cinema) < y(Veterinary Clinic) < y(Train Station) < y(Bank) < y(Stadium) < y(Gallery) < y(Pharmacy) < y(Theater) < y(City Hall) < y(Gym) => y(Cinema) < y(Gym).
SE(Cinema,Gym) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Courthouse) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Courthouse) < x(Gym).
NOT SE(Courthouse,Gym) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Restaurant) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Restaurant) < x(Gym).
NOT SE(Restaurant,Gym) in every valid arrangement.
[p14, p1] x(Clinic) < x(Post Office) < x(Gym) => x(Clinic) < x(Gym).
NOT SE(Clinic,Gym) in every valid arrangement.
[p2] x(Gym) < x(Museum).
[p2] y(Museum) < y(Gym).
SE(Museum,Gym) in every valid arrangement.
[p3] x(Cafe) < x(Gym).
NOT SE(Cafe,Gym) in every valid arrangement.
Fixed qualifying set={Hotel, Cinema, Museum}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Bakery,Gym) & NW(Courthouse,Gym) & NW(Restaurant,Gym) & NW(Clinic,Gym) & NW(Cafe,Gym). SE(Hotel,Gym) & SE(Cinema,Gym) & SE(Museum,Gym). Qualifying objects = {Cinema, Hotel, Museum}. count=3.

The claim that count=8 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The displayed entailed-answer set is {3}.

Answer: D

## production-b2758-s5-count-m0-exhaustive-entailed

Categories: contract-exhaustive-entailed.
Split: extrapolation-test; contract: exhaustive-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: exhaustive-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all A-D values only if they cover the complete nonempty set of individually entailed domain answers. Select F if any entailed answer is missing or no domain answer is entailed. Non-entailed answers outside the menu do not make coverage incomplete.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Hotel, Park, Gas Station, Post Office, Fire Station, Restaurant, Stadium, Supermarket, Hospital, Clinic, Train Station, Gym, High School, Bakery, Cinema, Zoo, Museum, Police Station, Courthouse, Cafe, Bank, Gallery, Theater, Florist, Bookstore, Grocery Store, City Hall, Veterinary Clinic, Library, Pharmacy
Query: count Southeast Gym
Eligible: Bakery, Hotel, Cinema, Courthouse, Restaurant, Clinic, Museum, Cafe
Consider a map with multiple objects:
Gym is to the Southeast of Post Office.
Gym is to the Northwest of Museum.
Cafe is to the Northwest of Gym.
Bakery is to the North of Police Station.
Restaurant is to the Northwest of Bookstore.
Train Station is to the Southeast of Bank.
Theater is to the Southeast of City Hall.
Bookstore is to the Southeast of Courthouse.
Fire Station is to the Southeast of Zoo.
Zoo is to the Southeast of Bakery.
Stadium is to the Northwest of Bank.
Veterinary Clinic is to the Northwest of Cinema.
Gallery is to the Northwest of Stadium.
Post Office is to the Southeast of Clinic.
Gym is to the Northwest of Hotel.
High School is to the Southeast of Grocery Store.
Park is to the Southeast of Library.
Restaurant is to the Southeast of Clinic.
Train Station is to the Northwest of Veterinary Clinic.
Pharmacy is to the Northwest of Gallery.
Hospital is to the Northwest of Florist.
Gym is to the Northwest of City Hall.
Gas Station is to the East of Train Station.
Florist is to the Northwest of Supermarket.
Pharmacy is to the Southeast of Theater.
Hospital is to the Southeast of Bookstore.
Fire Station is to the Northwest of Post Office.
Zoo is to the Southeast of Supermarket.
Question: Which exact counts of eligible objects southeast of Gym are entailed? Select the complete entailed-count set.
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. No complete entailed answer can be formed from A-D.

### Natural

The x-coordinates increase strictly along Bakery, Zoo, Fire Station, Post Office and Gym (premise sources: 10, 9, 27 and 1). So Gym has a greater x-coordinate than Bakery.
Bakery never qualifies for the queried relation.
Hotel has a greater x-coordinate than Gym (premise sources: 15).
Gym has a greater y-coordinate than Hotel (premise sources: 15).
Hotel always qualifies for the queried relation.
The x-coordinates increase strictly along Gym, City Hall, Theater, Pharmacy, Gallery, Stadium, Bank, Train Station, Veterinary Clinic and Cinema (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Cinema has a greater x-coordinate than Gym.
The y-coordinates increase strictly along Cinema, Veterinary Clinic, Train Station, Bank, Stadium, Gallery, Pharmacy, Theater, City Hall and Gym (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Gym has a greater y-coordinate than Cinema.
Cinema always qualifies for the queried relation.
The x-coordinates increase strictly along Courthouse, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Courthouse.
Courthouse never qualifies for the queried relation.
The x-coordinates increase strictly along Restaurant, Bookstore, Hospital, Florist, Supermarket, Zoo, Fire Station, Post Office and Gym (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Gym has a greater x-coordinate than Restaurant.
Restaurant never qualifies for the queried relation.
The x-coordinates increase strictly along Clinic, Post Office and Gym (premise sources: 14 and 1). So Gym has a greater x-coordinate than Clinic.
Clinic never qualifies for the queried relation.
Museum has a greater x-coordinate than Gym (premise sources: 2).
Gym has a greater y-coordinate than Museum (premise sources: 2).
Museum always qualifies for the queried relation.
Gym has a greater x-coordinate than Cafe (premise sources: 3).
Cafe never qualifies for the queried relation.
The qualifying candidates are fixed: Hotel, Cinema and Museum. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bakery, Courthouse, Restaurant, Clinic and Cafe are northwest of Gym. Hotel, Cinema and Museum are southeast of Gym. Only Cinema, Hotel and Museum qualify in this arrangement. This gives a count of 3.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Bakery) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Bakery) < x(Gym).
NOT SE(Bakery,Gym) in every valid arrangement.
[p15] x(Gym) < x(Hotel).
[p15] y(Hotel) < y(Gym).
SE(Hotel,Gym) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Gym) < x(City Hall) < x(Theater) < x(Pharmacy) < x(Gallery) < x(Stadium) < x(Bank) < x(Train Station) < x(Veterinary Clinic) < x(Cinema) => x(Gym) < x(Cinema).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Cinema) < y(Veterinary Clinic) < y(Train Station) < y(Bank) < y(Stadium) < y(Gallery) < y(Pharmacy) < y(Theater) < y(City Hall) < y(Gym) => y(Cinema) < y(Gym).
SE(Cinema,Gym) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Courthouse) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Courthouse) < x(Gym).
NOT SE(Courthouse,Gym) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Restaurant) < x(Bookstore) < x(Hospital) < x(Florist) < x(Supermarket) < x(Zoo) < x(Fire Station) < x(Post Office) < x(Gym) => x(Restaurant) < x(Gym).
NOT SE(Restaurant,Gym) in every valid arrangement.
[p14, p1] x(Clinic) < x(Post Office) < x(Gym) => x(Clinic) < x(Gym).
NOT SE(Clinic,Gym) in every valid arrangement.
[p2] x(Gym) < x(Museum).
[p2] y(Museum) < y(Gym).
SE(Museum,Gym) in every valid arrangement.
[p3] x(Cafe) < x(Gym).
NOT SE(Cafe,Gym) in every valid arrangement.
Fixed qualifying set={Hotel, Cinema, Museum}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Bakery,Gym) & NW(Courthouse,Gym) & NW(Restaurant,Gym) & NW(Clinic,Gym) & NW(Cafe,Gym). SE(Hotel,Gym) & SE(Cinema,Gym) & SE(Museum,Gym). Qualifying objects = {Cinema, Hotel, Museum}. count=3.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The claim that count=8 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

## production-b416-s4-direction-m0-unique-entailed

Categories: query-direction, depth-5.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Cafe, Hospital, Florist, Stadium, Theater, Bakery, Park, Gas Station, Pharmacy, Cinema, Train Station, Bookstore, Grocery Store, High School, Gym, University, Clinic, Fire Station, Restaurant, Post Office, Police Station, Courthouse, Zoo, City Hall, Library, Hotel, Veterinary Clinic, Supermarket
Query: direction Restaurant Theater
Consider a map with multiple objects:
Fire Station is to the Northeast of Zoo.
Fire Station is to the Southwest of University.
Hotel is to the Southeast of Cafe.
Pharmacy is to the Northwest of Stadium.
Restaurant is to the Southwest of Supermarket.
High School is to the Northeast of Theater.
Park is to the Northeast of Post Office.
Theater is to the Southwest of Post Office.
Veterinary Clinic is to the Northeast of Post Office.
Grocery Store is to the Southwest of Library.
Clinic is to the Southwest of City Hall.
City Hall is to the Southwest of Hospital.
Courthouse is to the Southeast of Hotel.
Theater is to the Southeast of Supermarket.
Courthouse is to the Northeast of Restaurant.
Florist is to the Southwest of Cinema.
Bookstore is to the Northwest of Cafe.
City Hall is to the Southwest of Zoo.
Grocery Store is to the Northeast of Hospital.
High School is to the Southwest of Florist.
Train Station is to the Northeast of Park.
Gym is to the Northwest of Bakery.
Police Station is to the Northeast of Train Station.
Bookstore is to the Southeast of Theater.
Theater is to the Southwest of Clinic.
Gas Station is to the North of Grocery Store.
Question: Which single direction of Restaurant relative to Theater is entailed?
A. Northwest
B. Southeast
C. East
D. Southwest
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Restaurant, Supermarket and Theater (premise sources: 5 and 14). So Theater has a greater x-coordinate than Restaurant.
The y-coordinates increase strictly along Restaurant, Courthouse, Hotel, Cafe, Bookstore and Theater (premise sources: 15, 13, 3, 17 and 24). So Theater has a greater y-coordinate than Restaurant.
Therefore, Restaurant is southwest of Theater in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. From left to right, the order is Restaurant, Supermarket and Theater. From bottom to top, the order is Restaurant, Theater and Supermarket. So Restaurant is southwest of Theater.

The claim that Restaurant is north of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is northeast of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is east of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is southeast of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is south of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is west of Theater is false in Case 1, so it is not entailed.

The claim that Restaurant is northwest of Theater is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: D

### Symbolic

[p5, p14] x(Restaurant) < x(Supermarket) < x(Theater) => x(Restaurant) < x(Theater).
[p15, p13, p3, p17, p24] y(Restaurant) < y(Courthouse) < y(Hotel) < y(Cafe) < y(Bookstore) < y(Theater) => y(Restaurant) < y(Theater).
SW(Restaurant,Theater) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Restaurant) < x(Supermarket) < x(Theater). y(Restaurant) < y(Theater) < y(Supermarket). => SW(Restaurant,Theater).

The claim that N(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that NE(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that E(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that SE(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that S(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that W(Restaurant,Theater) is false in Case 1, so it is not entailed.

The claim that NW(Restaurant,Theater) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: D

## production-b416-s4-which-m0-unique-entailed

Categories: query-which.
Split: train; contract: unique-entailed; answer: A.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Cafe, Hospital, Florist, Stadium, Theater, Bakery, Park, Gas Station, Pharmacy, Cinema, Train Station, Bookstore, Grocery Store, High School, Gym, University, Clinic, Fire Station, Restaurant, Post Office, Police Station, Courthouse, Zoo, City Hall, Library, Hotel, Veterinary Clinic, Supermarket
Query: which Southwest Theater
Eligible: Police Station, Restaurant, Library, Cinema, Veterinary Clinic, University
Consider a map with multiple objects:
Fire Station is to the Northeast of Zoo.
Fire Station is to the Southwest of University.
Hotel is to the Southeast of Cafe.
Pharmacy is to the Northwest of Stadium.
Restaurant is to the Southwest of Supermarket.
High School is to the Northeast of Theater.
Park is to the Northeast of Post Office.
Theater is to the Southwest of Post Office.
Veterinary Clinic is to the Northeast of Post Office.
Grocery Store is to the Southwest of Library.
Clinic is to the Southwest of City Hall.
City Hall is to the Southwest of Hospital.
Courthouse is to the Southeast of Hotel.
Theater is to the Southeast of Supermarket.
Courthouse is to the Northeast of Restaurant.
Florist is to the Southwest of Cinema.
Bookstore is to the Northwest of Cafe.
City Hall is to the Southwest of Zoo.
Grocery Store is to the Northeast of Hospital.
High School is to the Southwest of Florist.
Train Station is to the Northeast of Park.
Gym is to the Northwest of Bakery.
Police Station is to the Northeast of Train Station.
Bookstore is to the Southeast of Theater.
Theater is to the Southwest of Clinic.
Gas Station is to the North of Grocery Store.
Question: Which single eligible candidate is individually guaranteed to be southwest of Theater?
A. Restaurant
B. Veterinary Clinic
C. Police Station
D. Cinema
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Restaurant, Supermarket and Theater (premise sources: 5 and 14). So Theater has a greater x-coordinate than Restaurant.
The y-coordinates increase strictly along Restaurant, Courthouse, Hotel, Cafe, Bookstore and Theater (premise sources: 15, 13, 3, 17 and 24). So Theater has a greater y-coordinate than Restaurant.
Therefore, Restaurant is southwest of Theater in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Police Station, Library, Cinema, Veterinary Clinic and University are northeast of Theater. Restaurant is southwest of Theater. Only Restaurant qualifies in this arrangement.

The claim that Police Station is southwest of Theater is false in Case 1, so it is not entailed.

The claim that Library is southwest of Theater is false in Case 1, so it is not entailed.

The claim that Cinema is southwest of Theater is false in Case 1, so it is not entailed.

The claim that Veterinary Clinic is southwest of Theater is false in Case 1, so it is not entailed.

The claim that University is southwest of Theater is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Restaurant}.

Answer: A

### Symbolic

[p5, p14] x(Restaurant) < x(Supermarket) < x(Theater) => x(Restaurant) < x(Theater).
[p15, p13, p3, p17, p24] y(Restaurant) < y(Courthouse) < y(Hotel) < y(Cafe) < y(Bookstore) < y(Theater) => y(Restaurant) < y(Theater).
SW(Restaurant,Theater) in every valid arrangement.

Case 1 (counterexample satisfying all premises): NE(Police Station,Theater) & NE(Library,Theater) & NE(Cinema,Theater) & NE(Veterinary Clinic,Theater) & NE(University,Theater). SW(Restaurant,Theater). Qualifying objects = {Restaurant}.

The claim that SW(Police Station,Theater) is false in Case 1, so it is not entailed.

The claim that SW(Library,Theater) is false in Case 1, so it is not entailed.

The claim that SW(Cinema,Theater) is false in Case 1, so it is not entailed.

The claim that SW(Veterinary Clinic,Theater) is false in Case 1, so it is not entailed.

The claim that SW(University,Theater) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Restaurant}.

Answer: A

## production-b1953-s6-direction-m0-unique-entailed

Categories: depth-3.
Split: in-range-test; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Fire Station, Supermarket, Theater, Gallery, Gas Station, City Hall, Grocery Store, Restaurant, Post Office, Museum, Florist, Hotel, Gym, Hospital, Police Station, Bookstore, Park, Pharmacy, Library, Veterinary Clinic, Cafe, Bakery
Query: direction Hospital Park
Consider a map with multiple objects:
Theater is to the South of Post Office.
Police Station is to the North of Theater.
Gym is to the Northeast of Fire Station.
Park is to the South of Cafe.
Park is to the South of Hotel.
Hospital is to the Southeast of Fire Station.
Veterinary Clinic is to the North of Park.
Park is to the South of Theater.
Bakery is to the South of Gallery.
Hotel is to the South of City Hall.
City Hall is to the South of Cafe.
Park is to the Northeast of Gym.
Post Office is to the South of Pharmacy.
Florist is to the South of Park.
Police Station is to the South of Gas Station.
Grocery Store is to the South of Bookstore.
Fire Station is to the Southeast of Restaurant.
Florist is to the South of Hospital.
Supermarket is to the North of Theater.
Library is to the North of Museum.
Museum is to the North of Theater.
Question: Which single direction of Hospital relative to Park is entailed?
A. Northwest
B. East
C. South
D. North
E. Cannot be determined
F. None of the above

### Natural

Hospital, Florist and Park share the same x-coordinate (premise sources: 18 and 14).
The y-coordinates increase strictly along Hospital, Fire Station, Gym and Park (premise sources: 6, 3 and 12). So Park has a greater y-coordinate than Hospital.
Therefore, Hospital is south of Park in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Florist, Hospital and Park share the same horizontal position. From bottom to top, the order is Florist, Hospital and Park. So Hospital is south of Park.

The claim that Hospital is north of Park is false in Case 1, so it is not entailed.

The claim that Hospital is northeast of Park is false in Case 1, so it is not entailed.

The claim that Hospital is east of Park is false in Case 1, so it is not entailed.

The claim that Hospital is southeast of Park is false in Case 1, so it is not entailed.

The claim that Hospital is southwest of Park is false in Case 1, so it is not entailed.

The claim that Hospital is west of Park is false in Case 1, so it is not entailed.

The claim that Hospital is northwest of Park is false in Case 1, so it is not entailed.

The complete entailed-answer set is {South}.

Answer: C

### Symbolic

[p18, p14] x(Hospital) = x(Florist) = x(Park) => x(Hospital) = x(Park).
[p6, p3, p12] y(Hospital) < y(Fire Station) < y(Gym) < y(Park) => y(Hospital) < y(Park).
S(Hospital,Park) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Florist) = x(Hospital) = x(Park). y(Florist) < y(Hospital) < y(Park). => S(Hospital,Park).

The claim that N(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that NE(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that E(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that SE(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that SW(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that W(Hospital,Park) is false in Case 1, so it is not entailed.

The claim that NW(Hospital,Park) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {South}.

Answer: C

## production-b1492-s0-count-m0-unique-entailed

Categories: depth-6.
Split: dev; contract: unique-entailed; answer: A.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Florist, Bookstore, Veterinary Clinic, Supermarket, Courthouse, Pharmacy, Gym, Hotel, Police Station, University, Library, Train Station, High School, Post Office, Clinic, Fire Station, Park, Hospital, Bank, Theater
Query: count South 'Police Station'
Eligible: Library, Hotel, High School, Park, Bookstore
Consider a map with multiple objects:
Veterinary Clinic is to the North of Bank.
Clinic is to the South of Train Station.
Post Office is to the North of Veterinary Clinic.
Bookstore is to the North of Supermarket.
Courthouse is to the South of Library.
Park is to the North of Library.
Clinic is to the North of Post Office.
Supermarket is to the South of University.
High School is to the South of Police Station.
Courthouse is to the North of Police Station.
Police Station is to the North of Train Station.
Florist is to the East of Hospital.
Gym is to the North of University.
Fire Station is to the Northeast of Gym.
Bank is to the North of Hotel.
Courthouse is to the South of Theater.
Supermarket is to the North of Theater.
Gym is to the North of Pharmacy.
Park is to the North of Gym.
Question: What exact count of eligible objects south of Police Station is guaranteed?
A. 2
B. 1
C. 4
D. 0
E. Cannot be determined
F. None of the above

### Natural

The y-coordinates increase strictly along Police Station, Courthouse and Library (premise sources: 10 and 5). So Library has a greater y-coordinate than Police Station.
Library never qualifies for the queried relation.
Hotel, Bank, Veterinary Clinic, Post Office, Clinic, Train Station and Police Station share the same x-coordinate (premise sources: 15, 1, 3, 7, 2 and 11).
The y-coordinates increase strictly along Hotel, Bank, Veterinary Clinic, Post Office, Clinic, Train Station and Police Station (premise sources: 15, 1, 3, 7, 2 and 11). So Police Station has a greater y-coordinate than Hotel.
Hotel always qualifies for the queried relation.
High School and Police Station share the same x-coordinate (premise sources: 9).
Police Station has a greater y-coordinate than High School (premise sources: 9).
High School always qualifies for the queried relation.
The y-coordinates increase strictly along Police Station, Courthouse, Library and Park (premise sources: 10, 5 and 6). So Park has a greater y-coordinate than Police Station.
Park never qualifies for the queried relation.
The y-coordinates increase strictly along Police Station, Courthouse, Theater, Supermarket and Bookstore (premise sources: 10, 16, 17 and 4). So Bookstore has a greater y-coordinate than Police Station.
Bookstore never qualifies for the queried relation.
The qualifying candidates are fixed: Hotel and High School. Exactly 2 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Library, Park and Bookstore are north of Police Station. Hotel and High School are south of Police Station. Only High School and Hotel qualify in this arrangement. This gives a count of 2.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: A

### Symbolic

[p10, p5] y(Police Station) < y(Courthouse) < y(Library) => y(Police Station) < y(Library).
NOT S(Library,Police Station) in every valid arrangement.
[p15, p1, p3, p7, p2, p11] x(Hotel) = x(Bank) = x(Veterinary Clinic) = x(Post Office) = x(Clinic) = x(Train Station) = x(Police Station) => x(Hotel) = x(Police Station).
[p15, p1, p3, p7, p2, p11] y(Hotel) < y(Bank) < y(Veterinary Clinic) < y(Post Office) < y(Clinic) < y(Train Station) < y(Police Station) => y(Hotel) < y(Police Station).
S(Hotel,Police Station) in every valid arrangement.
[p9] x(High School) = x(Police Station).
[p9] y(High School) < y(Police Station).
S(High School,Police Station) in every valid arrangement.
[p10, p5, p6] y(Police Station) < y(Courthouse) < y(Library) < y(Park) => y(Police Station) < y(Park).
NOT S(Park,Police Station) in every valid arrangement.
[p10, p16, p17, p4] y(Police Station) < y(Courthouse) < y(Theater) < y(Supermarket) < y(Bookstore) => y(Police Station) < y(Bookstore).
NOT S(Bookstore,Police Station) in every valid arrangement.
Fixed qualifying set={Hotel, High School}; count=2 in every valid arrangement.

Case 1 (counterexample satisfying all premises): N(Library,Police Station) & N(Park,Police Station) & N(Bookstore,Police Station). S(Hotel,Police Station) & S(High School,Police Station). Qualifying objects = {High School, Hotel}. count=2.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: A

## production-b724-s1-count-m0-unique-entailed

Categories: depth-7.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Police Station, Museum, Grocery Store, City Hall, Restaurant, Courthouse, Cinema, Florist, Supermarket, Park, Veterinary Clinic, Gallery, Gas Station, Gym, University, Library, Zoo, Hospital, Bakery, Theater, Hotel, Train Station, Cafe, Clinic, Pharmacy, Bookstore
Query: count Southwest Museum
Eligible: Gas Station, Park, Theater, Courthouse, Restaurant, University, Zoo
Consider a map with multiple objects:
Supermarket is to the Southeast of Train Station.
Clinic is to the Northwest of Veterinary Clinic.
Train Station is to the Southeast of Cinema.
Gym is to the Southwest of Bakery.
Hotel is to the Northeast of Restaurant.
Cinema is to the Southeast of Museum.
Gym is to the Northeast of Pharmacy.
Museum is to the Southeast of Hotel.
Museum is to the Southwest of Courthouse.
Florist is to the Northeast of Hospital.
Gallery is to the Southeast of Veterinary Clinic.
Park is to the Northeast of Florist.
Museum is to the Southwest of Hospital.
Courthouse is to the Northeast of Theater.
City Hall is to the Northeast of Florist.
City Hall is to the Southwest of Gas Station.
Cafe is to the Northeast of Library.
Clinic is to the Southeast of Supermarket.
Library is to the Southwest of Theater.
Zoo is to the Southwest of Museum.
University is to the Northeast of Bakery.
Restaurant is to the Southwest of Gallery.
Police Station is to the East of Bookstore.
Grocery Store is to the Northeast of Gym.
Hospital is to the Southwest of Library.
Cafe is to the Southwest of Pharmacy.
Question: What exact count of eligible objects southwest of Museum is guaranteed?
A. 3
B. 5
C. 6
D. 2
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Museum, Hospital, Florist, City Hall and Gas Station (premise sources: 13, 10, 15 and 16). So Gas Station has a greater x-coordinate than Museum.
Gas Station never qualifies for the queried relation.
The x-coordinates increase strictly along Museum, Hospital, Florist and Park (premise sources: 13, 10 and 12). So Park has a greater x-coordinate than Museum.
Park never qualifies for the queried relation.
The x-coordinates increase strictly along Museum, Hospital, Library and Theater (premise sources: 13, 25 and 19). So Theater has a greater x-coordinate than Museum.
Theater never qualifies for the queried relation.
Courthouse has a greater x-coordinate than Museum (premise sources: 9).
Courthouse never qualifies for the queried relation.
The x-coordinates increase strictly along Restaurant, Hotel and Museum (premise sources: 5 and 8). So Museum has a greater x-coordinate than Restaurant.
The y-coordinates increase strictly along Restaurant, Gallery, Veterinary Clinic, Clinic, Supermarket, Train Station, Cinema and Museum (premise sources: 22, 11, 2, 18, 1, 3 and 6). So Museum has a greater y-coordinate than Restaurant.
Restaurant always qualifies for the queried relation.
The x-coordinates increase strictly along Museum, Hospital, Library, Cafe, Pharmacy, Gym, Bakery and University (premise sources: 13, 25, 17, 26, 7, 4 and 21). So University has a greater x-coordinate than Museum.
University never qualifies for the queried relation.
Museum has a greater x-coordinate than Zoo (premise sources: 20).
Museum has a greater y-coordinate than Zoo (premise sources: 20).
Zoo always qualifies for the queried relation.
The qualifying candidates are fixed: Restaurant and Zoo. Exactly 2 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Gas Station, Park, Theater, Courthouse and University are northeast of Museum. Restaurant and Zoo are southwest of Museum. Only Restaurant and Zoo qualify in this arrangement. This gives a count of 2.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: D

### Symbolic

[p13, p10, p15, p16] x(Museum) < x(Hospital) < x(Florist) < x(City Hall) < x(Gas Station) => x(Museum) < x(Gas Station).
NOT SW(Gas Station,Museum) in every valid arrangement.
[p13, p10, p12] x(Museum) < x(Hospital) < x(Florist) < x(Park) => x(Museum) < x(Park).
NOT SW(Park,Museum) in every valid arrangement.
[p13, p25, p19] x(Museum) < x(Hospital) < x(Library) < x(Theater) => x(Museum) < x(Theater).
NOT SW(Theater,Museum) in every valid arrangement.
[p9] x(Museum) < x(Courthouse).
NOT SW(Courthouse,Museum) in every valid arrangement.
[p5, p8] x(Restaurant) < x(Hotel) < x(Museum) => x(Restaurant) < x(Museum).
[p22, p11, p2, p18, p1, p3, p6] y(Restaurant) < y(Gallery) < y(Veterinary Clinic) < y(Clinic) < y(Supermarket) < y(Train Station) < y(Cinema) < y(Museum) => y(Restaurant) < y(Museum).
SW(Restaurant,Museum) in every valid arrangement.
[p13, p25, p17, p26, p7, p4, p21] x(Museum) < x(Hospital) < x(Library) < x(Cafe) < x(Pharmacy) < x(Gym) < x(Bakery) < x(University) => x(Museum) < x(University).
NOT SW(University,Museum) in every valid arrangement.
[p20] x(Zoo) < x(Museum).
[p20] y(Zoo) < y(Museum).
SW(Zoo,Museum) in every valid arrangement.
Fixed qualifying set={Restaurant, Zoo}; count=2 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NE(Gas Station,Museum) & NE(Park,Museum) & NE(Theater,Museum) & NE(Courthouse,Museum) & NE(University,Museum). SW(Restaurant,Museum) & SW(Zoo,Museum). Qualifying objects = {Restaurant, Zoo}. count=2.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: D

## production-b994-s0-count-m0-unique-entailed

Categories: depth-2.
Split: dev; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Supermarket, Pharmacy, High School, Bakery, Restaurant, Bank, Library, Gym, Train Station, Zoo, Clinic, Park, Hotel, Courthouse, Cinema, Bookstore
Query: count Northeast Courthouse
Eligible: Bank, Zoo, Hotel, Clinic, Park, Pharmacy
Consider a map with multiple objects:
Park is to the Northeast of Hotel.
Restaurant is to the South of Gym.
Clinic is to the Northeast of Train Station.
High School is to the Southwest of Restaurant.
Bakery is to the West of Supermarket.
Bank is to the Southwest of Zoo.
Courthouse is to the Southwest of Train Station.
Restaurant is to the Southwest of Courthouse.
Hotel is to the Southwest of Library.
Restaurant is to the Northeast of Zoo.
Bookstore is to the Northeast of Library.
Pharmacy is to the Northeast of Park.
Bank is to the Southwest of Pharmacy.
Cinema is to the Northeast of Bank.
Cinema is to the Southwest of Courthouse.
Courthouse is to the Northeast of Library.
Courthouse is to the Northeast of Park.
Pharmacy is to the Southwest of Restaurant.
Question: What exact count of eligible objects northeast of Courthouse is guaranteed?
A. 3
B. 0
C. 2
D. 1
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Bank, Cinema and Courthouse (premise sources: 14 and 15). So Courthouse has a greater x-coordinate than Bank.
Bank never qualifies for the queried relation.
The x-coordinates increase strictly along Zoo, Restaurant and Courthouse (premise sources: 10 and 8). So Courthouse has a greater x-coordinate than Zoo.
Zoo never qualifies for the queried relation.
The x-coordinates increase strictly along Hotel, Park and Courthouse (premise sources: 1 and 17). So Courthouse has a greater x-coordinate than Hotel.
Hotel never qualifies for the queried relation.
The x-coordinates increase strictly along Courthouse, Train Station and Clinic (premise sources: 7 and 3). So Clinic has a greater x-coordinate than Courthouse.
The y-coordinates increase strictly along Courthouse, Train Station and Clinic (premise sources: 7 and 3). So Clinic has a greater y-coordinate than Courthouse.
Clinic always qualifies for the queried relation.
Courthouse has a greater x-coordinate than Park (premise sources: 17).
Park never qualifies for the queried relation.
The x-coordinates increase strictly along Pharmacy, Restaurant and Courthouse (premise sources: 18 and 8). So Courthouse has a greater x-coordinate than Pharmacy.
Pharmacy never qualifies for the queried relation.
The qualifying candidates are fixed: Clinic. Exactly 1 qualifies in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bank, Zoo, Hotel, Park and Pharmacy are southwest of Courthouse. Clinic is northeast of Courthouse. Only Clinic qualifies in this arrangement. This gives a count of 1.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: D

### Symbolic

[p14, p15] x(Bank) < x(Cinema) < x(Courthouse) => x(Bank) < x(Courthouse).
NOT NE(Bank,Courthouse) in every valid arrangement.
[p10, p8] x(Zoo) < x(Restaurant) < x(Courthouse) => x(Zoo) < x(Courthouse).
NOT NE(Zoo,Courthouse) in every valid arrangement.
[p1, p17] x(Hotel) < x(Park) < x(Courthouse) => x(Hotel) < x(Courthouse).
NOT NE(Hotel,Courthouse) in every valid arrangement.
[p7, p3] x(Courthouse) < x(Train Station) < x(Clinic) => x(Courthouse) < x(Clinic).
[p7, p3] y(Courthouse) < y(Train Station) < y(Clinic) => y(Courthouse) < y(Clinic).
NE(Clinic,Courthouse) in every valid arrangement.
[p17] x(Park) < x(Courthouse).
NOT NE(Park,Courthouse) in every valid arrangement.
[p18, p8] x(Pharmacy) < x(Restaurant) < x(Courthouse) => x(Pharmacy) < x(Courthouse).
NOT NE(Pharmacy,Courthouse) in every valid arrangement.
Fixed qualifying set={Clinic}; count=1 in every valid arrangement.

Case 1 (counterexample satisfying all premises): SW(Bank,Courthouse) & SW(Zoo,Courthouse) & SW(Hotel,Courthouse) & SW(Park,Courthouse) & SW(Pharmacy,Courthouse). NE(Clinic,Courthouse). Qualifying objects = {Clinic}. count=1.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: D

## production-b11-s7-count-m0-unique-entailed

Categories: depth-1.
Split: train; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: City Hall, Hospital, Bank, Supermarket, Clinic, Post Office, Park, Gas Station, Pharmacy, Bakery, Theater, Hotel, Restaurant, Gallery
Query: count South 'City Hall'
Eligible: Gallery, Park, Post Office, Restaurant, Bakery, Theater, Gas Station
Consider a map with multiple objects:
Park is to the North of Restaurant.
City Hall is to the South of Bakery.
Restaurant is to the North of City Hall.
Theater is to the South of Bakery.
Post Office is to the South of Bakery.
Gallery is to the West of Hotel.
Theater is to the North of City Hall.
Pharmacy is to the Northeast of Post Office.
Supermarket is to the Northwest of Bank.
City Hall is to the South of Gallery.
Post Office is to the North of City Hall.
Gas Station is to the North of City Hall.
Restaurant is to the South of Gas Station.
Clinic is to the East of Hospital.
City Hall is to the South of Park.
Question: What exact count of eligible objects south of City Hall is guaranteed?
A. 7
B. 5
C. 0
D. 1
E. Cannot be determined
F. None of the above

### Natural

Gallery has a greater y-coordinate than City Hall (premise sources: 10).
Gallery never qualifies for the queried relation.
Park has a greater y-coordinate than City Hall (premise sources: 15).
Park never qualifies for the queried relation.
Post Office has a greater y-coordinate than City Hall (premise sources: 11).
Post Office never qualifies for the queried relation.
Restaurant has a greater y-coordinate than City Hall (premise sources: 3).
Restaurant never qualifies for the queried relation.
Bakery has a greater y-coordinate than City Hall (premise sources: 2).
Bakery never qualifies for the queried relation.
Theater has a greater y-coordinate than City Hall (premise sources: 7).
Theater never qualifies for the queried relation.
Gas Station has a greater y-coordinate than City Hall (premise sources: 12).
Gas Station never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Gallery, Park, Post Office, Restaurant, Bakery, Theater and Gas Station are north of City Hall. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

### Symbolic

[p10] y(City Hall) < y(Gallery).
NOT S(Gallery,City Hall) in every valid arrangement.
[p15] y(City Hall) < y(Park).
NOT S(Park,City Hall) in every valid arrangement.
[p11] y(City Hall) < y(Post Office).
NOT S(Post Office,City Hall) in every valid arrangement.
[p3] y(City Hall) < y(Restaurant).
NOT S(Restaurant,City Hall) in every valid arrangement.
[p2] y(City Hall) < y(Bakery).
NOT S(Bakery,City Hall) in every valid arrangement.
[p7] y(City Hall) < y(Theater).
NOT S(Theater,City Hall) in every valid arrangement.
[p12] y(City Hall) < y(Gas Station).
NOT S(Gas Station,City Hall) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

Case 1 (counterexample satisfying all premises): N(Gallery,City Hall) & N(Park,City Hall) & N(Post Office,City Hall) & N(Restaurant,City Hall) & N(Bakery,City Hall) & N(Theater,City Hall) & N(Gas Station,City Hall). Qualifying objects = {}. count=0.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

## production-b534-s15-direction-m1-unique-entailed

Categories: escape-unique-unlisted, omission.
Split: train; contract: unique-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Bookstore, Post Office, Supermarket, Hotel, Police Station, Bank, Gallery, Clinic, Veterinary Clinic, Florist, Stadium, Theater, Train Station, Courthouse, Cinema, Restaurant, Gym, Gas Station, University, Fire Station, Cafe, City Hall, Library, High School, Zoo, Museum, Hospital, Grocery Store, Pharmacy
Query: direction Supermarket Stadium
Consider a map with multiple objects:
Veterinary Clinic is to the South of Hotel.
City Hall is to the Northeast of Hospital.
Gas Station is to the Northeast of Theater.
Bank is to the Northwest of Restaurant.
City Hall is to the Southwest of Cinema.
Veterinary Clinic is to the North of Post Office.
University is to the South of Museum.
Florist is to the South of Library.
Gallery is to the South of Courthouse.
Stadium is to the South of Grocery Store.
Fire Station is to the North of Restaurant.
Florist is to the South of Zoo.
Cinema is to the Southwest of Stadium.
Gas Station is to the Southwest of Hospital.
Cafe is to the North of Bookstore.
Supermarket is to the North of Police Station.
Stadium is to the South of Gym.
University is to the North of Hotel.
Library is to the South of Restaurant.
Clinic is to the North of Florist.
Hotel is to the South of Bookstore.
Grocery Store is to the South of Post Office.
Cafe is to the North of Fire Station.
Supermarket is to the Southeast of Theater.
Pharmacy is to the Northeast of High School.
Florist is to the North of Grocery Store.
Police Station is to the South of Stadium.
Train Station is to the Northwest of Zoo.
Question: Which single direction of Supermarket relative to Stadium is entailed?
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Supermarket, Police Station and Stadium share the same x-coordinate (premise sources: 16 and 27).
The y-coordinates increase strictly along Supermarket, Theater, Gas Station, Hospital, City Hall, Cinema and Stadium (premise sources: 24, 3, 14, 2, 5 and 13). So Stadium has a greater y-coordinate than Supermarket.
Therefore, Supermarket is south of Stadium in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

### Symbolic

[p16, p27] x(Supermarket) = x(Police Station) = x(Stadium) => x(Supermarket) = x(Stadium).
[p24, p3, p14, p2, p5, p13] y(Supermarket) < y(Theater) < y(Gas Station) < y(Hospital) < y(City Hall) < y(Cinema) < y(Stadium) => y(Supermarket) < y(Stadium).
S(Supermarket,Stadium) in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

## production-b534-s15-direction-m1-listed-entailed

Categories: escape-no-listed-entailments.
Split: train; contract: listed-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: listed-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all displayed A-D values that are entailed. Exclude every displayed value that is not entailed, including claims that could hold but are not guaranteed. Select F if no displayed value is entailed.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Bookstore, Post Office, Supermarket, Hotel, Police Station, Bank, Gallery, Clinic, Veterinary Clinic, Florist, Stadium, Theater, Train Station, Courthouse, Cinema, Restaurant, Gym, Gas Station, University, Fire Station, Cafe, City Hall, Library, High School, Zoo, Museum, Hospital, Grocery Store, Pharmacy
Query: direction Supermarket Stadium
Consider a map with multiple objects:
Veterinary Clinic is to the South of Hotel.
City Hall is to the Northeast of Hospital.
Gas Station is to the Northeast of Theater.
Bank is to the Northwest of Restaurant.
City Hall is to the Southwest of Cinema.
Veterinary Clinic is to the North of Post Office.
University is to the South of Museum.
Florist is to the South of Library.
Gallery is to the South of Courthouse.
Stadium is to the South of Grocery Store.
Fire Station is to the North of Restaurant.
Florist is to the South of Zoo.
Cinema is to the Southwest of Stadium.
Gas Station is to the Southwest of Hospital.
Cafe is to the North of Bookstore.
Supermarket is to the North of Police Station.
Stadium is to the South of Gym.
University is to the North of Hotel.
Library is to the South of Restaurant.
Clinic is to the North of Florist.
Hotel is to the South of Bookstore.
Grocery Store is to the South of Post Office.
Cafe is to the North of Fire Station.
Supermarket is to the Southeast of Theater.
Pharmacy is to the Northeast of High School.
Florist is to the North of Grocery Store.
Police Station is to the South of Stadium.
Train Station is to the Northwest of Zoo.
Question: Which listed directions of Supermarket relative to Stadium are entailed? Select all that apply.
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Police Station, Stadium and Supermarket share the same horizontal position. From bottom to top, the order is Police Station, Supermarket and Stadium. So Supermarket is south of Stadium.

The claim that Supermarket is west of Stadium is false in Case 1, so it is not entailed.

The claim that Supermarket is northeast of Stadium is false in Case 1, so it is not entailed.

The claim that Supermarket is southwest of Stadium is false in Case 1, so it is not entailed.

The claim that Supermarket is northwest of Stadium is false in Case 1, so it is not entailed.

No displayed answer is entailed.

Answer: F

### Symbolic

Case 1 (counterexample satisfying all premises): x(Police Station) = x(Stadium) = x(Supermarket). y(Police Station) < y(Supermarket) < y(Stadium). => S(Supermarket,Stadium).

The claim that W(Supermarket,Stadium) is false in Case 1, so it is not entailed.

The claim that NE(Supermarket,Stadium) is false in Case 1, so it is not entailed.

The claim that SW(Supermarket,Stadium) is false in Case 1, so it is not entailed.

The claim that NW(Supermarket,Stadium) is false in Case 1, so it is not entailed.

No displayed answer is entailed.

Answer: F

## production-b534-s15-direction-m1-exhaustive-entailed

Categories: escape-incomplete-entailed-coverage.
Split: train; contract: exhaustive-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: exhaustive-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all A-D values only if they cover the complete nonempty set of individually entailed domain answers. Select F if any entailed answer is missing or no domain answer is entailed. Non-entailed answers outside the menu do not make coverage incomplete.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Bookstore, Post Office, Supermarket, Hotel, Police Station, Bank, Gallery, Clinic, Veterinary Clinic, Florist, Stadium, Theater, Train Station, Courthouse, Cinema, Restaurant, Gym, Gas Station, University, Fire Station, Cafe, City Hall, Library, High School, Zoo, Museum, Hospital, Grocery Store, Pharmacy
Query: direction Supermarket Stadium
Consider a map with multiple objects:
Veterinary Clinic is to the South of Hotel.
City Hall is to the Northeast of Hospital.
Gas Station is to the Northeast of Theater.
Bank is to the Northwest of Restaurant.
City Hall is to the Southwest of Cinema.
Veterinary Clinic is to the North of Post Office.
University is to the South of Museum.
Florist is to the South of Library.
Gallery is to the South of Courthouse.
Stadium is to the South of Grocery Store.
Fire Station is to the North of Restaurant.
Florist is to the South of Zoo.
Cinema is to the Southwest of Stadium.
Gas Station is to the Southwest of Hospital.
Cafe is to the North of Bookstore.
Supermarket is to the North of Police Station.
Stadium is to the South of Gym.
University is to the North of Hotel.
Library is to the South of Restaurant.
Clinic is to the North of Florist.
Hotel is to the South of Bookstore.
Grocery Store is to the South of Post Office.
Cafe is to the North of Fire Station.
Supermarket is to the Southeast of Theater.
Pharmacy is to the Northeast of High School.
Florist is to the North of Grocery Store.
Police Station is to the South of Stadium.
Train Station is to the Northwest of Zoo.
Question: Select the complete set of individually entailed directions of Supermarket relative to Stadium.
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. No complete entailed answer can be formed from A-D.

### Natural

Supermarket, Police Station and Stadium share the same x-coordinate (premise sources: 16 and 27).
The y-coordinates increase strictly along Supermarket, Theater, Gas Station, Hospital, City Hall, Cinema and Stadium (premise sources: 24, 3, 14, 2, 5 and 13). So Stadium has a greater y-coordinate than Supermarket.
Therefore, Supermarket is south of Stadium in every valid arrangement.

An entailed answer is missing from A-D, so the complete entailed set cannot be represented.

Answer: F

### Symbolic

[p16, p27] x(Supermarket) = x(Police Station) = x(Stadium) => x(Supermarket) = x(Stadium).
[p24, p3, p14, p2, p5, p13] y(Supermarket) < y(Theater) < y(Gas Station) < y(Hospital) < y(City Hall) < y(Cinema) < y(Stadium) => y(Supermarket) < y(Stadium).
S(Supermarket,Stadium) in every valid arrangement.

An entailed answer is missing from A-D, so the complete entailed set cannot be represented.

Answer: F

## production-b2554-s6-direction-m0-unique-entailed

Categories: depth-8.
Split: in-range-test; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Clinic, Train Station, Cinema, Zoo, Gallery, Library, Supermarket, Post Office, Restaurant, Gas Station, Grocery Store, Pharmacy, Hospital, Courthouse, High School, University, Hotel, City Hall, Theater, Stadium, Veterinary Clinic, Bookstore, Florist, Bakery, Police Station, Bank, Gym, Museum
Query: direction Pharmacy 'Grocery Store'
Consider a map with multiple objects:
High School is to the Southwest of Theater.
Clinic is to the North of Supermarket.
Hospital is to the Northeast of Gas Station.
Hospital is to the Northeast of Bakery.
Courthouse is to the Southwest of Gallery.
Restaurant is to the Northeast of Grocery Store.
Hospital is to the Northeast of Cinema.
Theater is to the Southwest of Cinema.
Pharmacy is to the South of Post Office.
Bank is to the Southwest of Library.
Gym is to the Southwest of Bakery.
Train Station is to the Southwest of Courthouse.
City Hall is to the Southwest of Bookstore.
Zoo is to the West of University.
Gas Station is to the Northeast of Grocery Store.
Police Station is to the South of Florist.
Florist is to the West of Supermarket.
Hotel is to the North of Museum.
Museum is to the East of Clinic.
Hotel is to the West of Grocery Store.
Courthouse is to the West of Stadium.
Police Station is to the East of Post Office.
Courthouse is to the Southwest of Gym.
Courthouse is to the Southwest of Bank.
Veterinary Clinic is to the Southwest of Cinema.
Train Station is to the Northeast of Grocery Store.
Restaurant is to the Southwest of Gallery.
High School is to the Northeast of Library.
Question: Which single direction of Pharmacy relative to Grocery Store is entailed?
A. Northwest
B. North
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above

### Natural

Following the x-coordinates, Pharmacy shares its x-coordinate with Post Office; Post Office has a smaller x-coordinate than Police Station; Police Station shares its x-coordinate with Florist; Florist has a smaller x-coordinate than Supermarket; Supermarket shares its x-coordinate with Clinic; Clinic has a smaller x-coordinate than Museum; Museum shares its x-coordinate with Hotel; Hotel has a smaller x-coordinate than Grocery Store (premise sources: 9, 22, 16, 17, 2, 19, 18 and 20). Therefore, Grocery Store has a greater x-coordinate than Pharmacy.
Following the y-coordinates, Pharmacy has a smaller y-coordinate than Post Office; Post Office shares its y-coordinate with Police Station; Police Station has a smaller y-coordinate than Florist; Florist shares its y-coordinate with Supermarket; Supermarket has a smaller y-coordinate than Clinic; Clinic shares its y-coordinate with Museum; Museum has a smaller y-coordinate than Hotel; Hotel shares its y-coordinate with Grocery Store (premise sources: 9, 22, 16, 17, 2, 19, 18 and 20). Therefore, Grocery Store has a greater y-coordinate than Pharmacy.
Therefore, Pharmacy is southwest of Grocery Store in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. From left to right, the order is Pharmacy and Grocery Store. From bottom to top, the order is Pharmacy and Grocery Store. So Pharmacy is southwest of Grocery Store.

The claim that Pharmacy is north of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is northeast of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is east of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is southeast of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is south of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is west of Grocery Store is false in Case 1, so it is not entailed.

The claim that Pharmacy is northwest of Grocery Store is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: C

### Symbolic

[p9, p22, p16, p17, p2, p19, p18, p20] x(Pharmacy) = x(Post Office) < x(Police Station) = x(Florist) < x(Supermarket) = x(Clinic) < x(Museum) = x(Hotel) < x(Grocery Store) => x(Pharmacy) < x(Grocery Store).
[p9, p22, p16, p17, p2, p19, p18, p20] y(Pharmacy) < y(Post Office) = y(Police Station) < y(Florist) = y(Supermarket) < y(Clinic) = y(Museum) < y(Hotel) = y(Grocery Store) => y(Pharmacy) < y(Grocery Store).
SW(Pharmacy,Grocery Store) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Pharmacy) < x(Grocery Store). y(Pharmacy) < y(Grocery Store). => SW(Pharmacy,Grocery Store).

The claim that N(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that NE(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that E(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that SE(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that S(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that W(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The claim that NW(Pharmacy,Grocery Store) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: C

## production-b328-s4-count-m0-unique-entailed

Categories: depth-4.
Split: train; contract: unique-entailed; answer: B.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Zoo, Hotel, Courthouse, Police Station, Bank, Restaurant, Fire Station, Bookstore, Library, Supermarket, Train Station, Grocery Store, University, Bakery, Veterinary Clinic, Florist, Cinema, Theater, Park
Query: count Southeast Library
Eligible: Theater, Supermarket, Bakery, Hotel, Park
Consider a map with multiple objects:
Bakery is to the Northwest of Bank.
Courthouse is to the East of Hotel.
Hotel is to the Northwest of Grocery Store.
Zoo is to the West of University.
Restaurant is to the Southeast of Florist.
Hotel is to the Southeast of Bakery.
Police Station is to the North of Train Station.
Grocery Store is to the Northwest of Florist.
Bookstore is to the Southwest of Fire Station.
Cinema is to the East of Bank.
Library is to the Southeast of Supermarket.
Theater is to the Southeast of Hotel.
Park is to the East of Fire Station.
Florist is to the Southeast of Bank.
Restaurant is to the Northwest of Library.
Hotel is to the Southeast of Supermarket.
Theater is to the Northwest of Veterinary Clinic.
University is to the North of Fire Station.
Veterinary Clinic is to the Northwest of Library.
Library is to the North of Zoo.
Question: What exact count of eligible objects southeast of Library is guaranteed?
A. 3
B. 1
C. 2
D. 0
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Theater, Veterinary Clinic and Library (premise sources: 17 and 19). So Library has a greater x-coordinate than Theater.
Theater never qualifies for the queried relation.
Library has a greater x-coordinate than Supermarket (premise sources: 11).
Supermarket never qualifies for the queried relation.
The x-coordinates increase strictly along Bakery, Hotel, Theater, Veterinary Clinic and Library (premise sources: 6, 12, 17 and 19). So Library has a greater x-coordinate than Bakery.
Bakery never qualifies for the queried relation.
The x-coordinates increase strictly along Hotel, Theater, Veterinary Clinic and Library (premise sources: 12, 17 and 19). So Library has a greater x-coordinate than Hotel.
Hotel never qualifies for the queried relation.
Following the x-coordinates, Library shares its x-coordinate with Zoo; Zoo has a smaller x-coordinate than University; University shares its x-coordinate with Fire Station; Fire Station has a smaller x-coordinate than Park (premise sources: 20, 4, 18 and 13). Therefore, Park has a greater x-coordinate than Library.
Following the y-coordinates, Park shares its y-coordinate with Fire Station; Fire Station has a smaller y-coordinate than University; University shares its y-coordinate with Zoo; Zoo has a smaller y-coordinate than Library (premise sources: 13, 18, 4 and 20). Therefore, Library has a greater y-coordinate than Park.
Park always qualifies for the queried relation.
The qualifying candidates are fixed: Park. Exactly 1 qualifies in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Theater, Supermarket, Bakery and Hotel are northwest of Library. Park is southeast of Library. Only Park qualifies in this arrangement. This gives a count of 1.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: B

### Symbolic

[p17, p19] x(Theater) < x(Veterinary Clinic) < x(Library) => x(Theater) < x(Library).
NOT SE(Theater,Library) in every valid arrangement.
[p11] x(Supermarket) < x(Library).
NOT SE(Supermarket,Library) in every valid arrangement.
[p6, p12, p17, p19] x(Bakery) < x(Hotel) < x(Theater) < x(Veterinary Clinic) < x(Library) => x(Bakery) < x(Library).
NOT SE(Bakery,Library) in every valid arrangement.
[p12, p17, p19] x(Hotel) < x(Theater) < x(Veterinary Clinic) < x(Library) => x(Hotel) < x(Library).
NOT SE(Hotel,Library) in every valid arrangement.
[p20, p4, p18, p13] x(Library) = x(Zoo) < x(University) = x(Fire Station) < x(Park) => x(Library) < x(Park).
[p13, p18, p4, p20] y(Park) = y(Fire Station) < y(University) = y(Zoo) < y(Library) => y(Park) < y(Library).
SE(Park,Library) in every valid arrangement.
Fixed qualifying set={Park}; count=1 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Theater,Library) & NW(Supermarket,Library) & NW(Bakery,Library) & NW(Hotel,Library). SE(Park,Library). Qualifying objects = {Park}. count=1.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: B

## production-b2905-s118-direction-m0-unique-entailed

Categories: depth-None, escape-inconsistent.
Split: train; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Courthouse, Veterinary Clinic, Pharmacy, High School, Cinema, Park
Query: direction 'Veterinary Clinic' Cinema
Consider a map with multiple objects:
Cinema is to the South of Pharmacy.
Cinema is to the North of Park.
Park is to the South of Veterinary Clinic.
Courthouse is to the West of Park.
High School is to the Northwest of Cinema.
Veterinary Clinic is to the Northwest of Pharmacy.
Question: Which single direction of Veterinary Clinic relative to Cinema is entailed?
A. Southeast
B. North
C. East
D. Northeast
E. Cannot be determined
F. None of the above

### Natural

From p6, Veterinary Clinic's x-coordinate is less than Pharmacy's x-coordinate.
From p1, Pharmacy's x-coordinate is equal to Cinema's x-coordinate.
From p2, Cinema's x-coordinate is equal to Park's x-coordinate.
From p3, Park's x-coordinate is equal to Veterinary Clinic's x-coordinate.
Composing this path gives Veterinary Clinic's x-coordinate less than Veterinary Clinic's x-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p2, p3, p6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

### Symbolic

p6: x(Veterinary Clinic) < x(Pharmacy)
p1: x(Pharmacy) = x(Cinema)
p2: x(Cinema) = x(Park)
p3: x(Park) = x(Veterinary Clinic)
Transitivity: x(Veterinary Clinic) < x(Veterinary Clinic)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p2, p3, p6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

## production-b2925-s2-which-m0-unique-entailed

Categories: escape-multiple-entailed.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Post Office, Restaurant, Stadium, Theater, Cafe, Hospital, Bank, Bakery, Pharmacy, Veterinary Clinic, Supermarket
Query: which South Hospital
Eligible: Supermarket, Post Office, Bakery, Theater, Pharmacy, Stadium, Cafe
Consider a map with multiple objects:
Veterinary Clinic is to the South of Bakery.
Hospital is to the South of Veterinary Clinic.
Veterinary Clinic is to the South of Stadium.
Pharmacy is to the South of Hospital.
Hospital is to the South of Cafe.
Hospital is to the South of Supermarket.
Theater is to the North of Cafe.
Hospital is to the North of Bank.
Theater is to the North of Veterinary Clinic.
Bank is to the North of Post Office.
Restaurant is to the South of Pharmacy.
Theater is to the North of Bakery.
Question: Which single eligible candidate is individually guaranteed to be south of Hospital?
A. Pharmacy
B. Post Office
C. Supermarket
D. Stadium
E. Cannot be determined
F. None of the above

### Natural

Post Office, Bank and Hospital share the same x-coordinate (premise sources: 10 and 8).
The y-coordinates increase strictly along Post Office, Bank and Hospital (premise sources: 10 and 8). So Hospital has a greater y-coordinate than Post Office.
Therefore, Post Office is south of Hospital in every valid arrangement.

Pharmacy and Hospital share the same x-coordinate (premise sources: 4).
Hospital has a greater y-coordinate than Pharmacy (premise sources: 4).
Therefore, Pharmacy is south of Hospital in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p10, p8] x(Post Office) = x(Bank) = x(Hospital) => x(Post Office) = x(Hospital).
[p10, p8] y(Post Office) < y(Bank) < y(Hospital) => y(Post Office) < y(Hospital).
S(Post Office,Hospital) in every valid arrangement.

[p4] x(Pharmacy) = x(Hospital).
[p4] y(Pharmacy) < y(Hospital).
S(Pharmacy,Hospital) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2925-s2-which-m0-unique-entailed-position1

Categories: position-variant.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Post Office, Restaurant, Stadium, Theater, Cafe, Hospital, Bank, Bakery, Pharmacy, Veterinary Clinic, Supermarket
Query: which South Hospital
Eligible: Supermarket, Post Office, Bakery, Theater, Pharmacy, Stadium, Cafe
Consider a map with multiple objects:
Veterinary Clinic is to the South of Bakery.
Hospital is to the South of Veterinary Clinic.
Veterinary Clinic is to the South of Stadium.
Pharmacy is to the South of Hospital.
Hospital is to the South of Cafe.
Hospital is to the South of Supermarket.
Theater is to the North of Cafe.
Hospital is to the North of Bank.
Theater is to the North of Veterinary Clinic.
Bank is to the North of Post Office.
Restaurant is to the South of Pharmacy.
Theater is to the North of Bakery.
Question: Which single eligible candidate is individually guaranteed to be south of Hospital?
A. Pharmacy
B. Supermarket
C. Post Office
D. Stadium
E. Cannot be determined
F. None of the above

### Natural

Post Office, Bank and Hospital share the same x-coordinate (premise sources: 10 and 8).
The y-coordinates increase strictly along Post Office, Bank and Hospital (premise sources: 10 and 8). So Hospital has a greater y-coordinate than Post Office.
Therefore, Post Office is south of Hospital in every valid arrangement.

Pharmacy and Hospital share the same x-coordinate (premise sources: 4).
Hospital has a greater y-coordinate than Pharmacy (premise sources: 4).
Therefore, Pharmacy is south of Hospital in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p10, p8] x(Post Office) = x(Bank) = x(Hospital) => x(Post Office) = x(Hospital).
[p10, p8] y(Post Office) < y(Bank) < y(Hospital) => y(Post Office) < y(Hospital).
S(Post Office,Hospital) in every valid arrangement.

[p4] x(Pharmacy) = x(Hospital).
[p4] y(Pharmacy) < y(Hospital).
S(Pharmacy,Hospital) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2922-s132-direction-m0-unique-entailed

Categories: escape-no-entailed-answer.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Gym, Clinic, Courthouse, Florist, Grocery Store, Hotel, Library, Hospital
Query: direction Florist Courthouse
Consider a map with multiple objects:
Library is to the East of Gym.
Courthouse is to the South of Library.
Hospital is to the Southeast of Library.
Grocery Store is to the Northeast of Gym.
Florist is to the Southeast of Library.
Hotel is to the Southwest of Grocery Store.
Courthouse is to the Northeast of Hotel.
Question: Which single direction of Florist relative to Courthouse is entailed?
A. Southeast
B. Northeast
C. North
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Courthouse and Library lie to the left of Florist. Courthouse and Library share the same horizontal position. From bottom to top, the order is Florist, Courthouse and Library. So Florist is southeast of Courthouse.

Case 2: A counterexample arrangement satisfying all premises is the following. Courthouse and Library lie to the left of Florist. Courthouse and Library share the same horizontal position. From bottom to top, the order is Courthouse, Florist and Library. So Florist is northeast of Courthouse.

The claim that Florist is north of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is northeast of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is east of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is southeast of Courthouse is false in Case 2, so it is not entailed.

The claim that Florist is south of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is southwest of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is west of Courthouse is false in Case 1, so it is not entailed.

The claim that Florist is northwest of Courthouse is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

### Symbolic

Case 1 (counterexample satisfying all premises): x(Courthouse) = x(Library) < x(Florist). y(Florist) < y(Courthouse) < y(Library). => SE(Florist,Courthouse).

Case 2 (counterexample satisfying all premises): x(Courthouse) = x(Library) < x(Florist). y(Courthouse) < y(Florist) < y(Library). => NE(Florist,Courthouse).

The claim that N(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that NE(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that E(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that SE(Florist,Courthouse) is false in Case 2, so it is not entailed.

The claim that S(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that SW(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that W(Florist,Courthouse) is false in Case 1, so it is not entailed.

The claim that NW(Florist,Courthouse) is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

## production-b2907-s32-which-m0-unique-entailed

Categories: three-entailed.
Split: train; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Florist, City Hall, Post Office, Bank, University, Veterinary Clinic, Gas Station, Gym, Fire Station, Pharmacy, Theater, Train Station, Stadium, High School
Query: which Northeast Bank
Eligible: Gym, Theater, High School, Train Station, Gas Station, Fire Station, University
Consider a map with multiple objects:
University is to the Southwest of Bank.
Fire Station is to the Southwest of Stadium.
Bank is to the Northeast of Stadium.
Bank is to the Southwest of Theater.
High School is to the Northeast of Bank.
Veterinary Clinic is to the Northeast of Gas Station.
Bank is to the Northeast of Train Station.
University is to the Southwest of Train Station.
Post Office is to the West of Florist.
Bank is to the Northeast of Veterinary Clinic.
Gas Station is to the Northwest of City Hall.
Pharmacy is to the Southwest of Gym.
Pharmacy is to the Northeast of Bank.
Question: Which single eligible candidate is individually guaranteed to be northeast of Bank?
A. Theater
B. Gas Station
C. Gym
D. High School
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Bank, Pharmacy and Gym (premise sources: 13 and 12). So Gym has a greater x-coordinate than Bank.
The y-coordinates increase strictly along Bank, Pharmacy and Gym (premise sources: 13 and 12). So Gym has a greater y-coordinate than Bank.
Therefore, Gym is northeast of Bank in every valid arrangement.

Theater has a greater x-coordinate than Bank (premise sources: 4).
Theater has a greater y-coordinate than Bank (premise sources: 4).
Therefore, Theater is northeast of Bank in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p13, p12] x(Bank) < x(Pharmacy) < x(Gym) => x(Bank) < x(Gym).
[p13, p12] y(Bank) < y(Pharmacy) < y(Gym) => y(Bank) < y(Gym).
NE(Gym,Bank) in every valid arrangement.

[p4] x(Bank) < x(Theater).
[p4] y(Bank) < y(Theater).
NE(Theater,Bank) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2816-s0-count-m0-unique-entailed

Categories: depth-10.
Split: extrapolation-test; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Police Station, University, Gym, Grocery Store, Hospital, Cinema, Bakery, Bookstore, High School, Clinic, Gallery, Bank, City Hall, Veterinary Clinic, Supermarket, Hotel, Theater, Park, Museum, Pharmacy, Florist, Cafe, Stadium, Courthouse, Library, Zoo, Post Office, Train Station, Gas Station
Query: count East Bookstore
Eligible: Bakery, Park, Supermarket, Courthouse, High School, Florist
Consider a map with multiple objects:
Gym is to the East of Park.
Grocery Store is to the West of Hospital.
Stadium is to the East of Courthouse.
Museum is to the West of Theater.
Bank is to the West of Gallery.
Hotel is to the East of High School.
Hospital is to the West of Post Office.
Cinema is to the West of Clinic.
Zoo is to the South of City Hall.
Museum is to the East of Florist.
Stadium is to the East of Museum.
Gas Station is to the West of Police Station.
Veterinary Clinic is to the West of Bookstore.
Bank is to the East of Library.
High School is to the West of Bookstore.
City Hall is to the East of Post Office.
Bank is to the East of Train Station.
City Hall is to the West of Cinema.
Gallery is to the West of Bookstore.
Pharmacy is to the West of Museum.
Clinic is to the West of Veterinary Clinic.
University is to the Northeast of Cafe.
Gas Station is to the East of Bakery.
Library is to the East of Stadium.
Supermarket is to the West of Pharmacy.
Supermarket is to the East of Park.
Grocery Store is to the East of Police Station.
Gym is to the West of Train Station.
Question: What exact count of eligible objects east of Bookstore is guaranteed?
A. 4
B. 3
C. 5
D. 0
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Bakery, Gas Station, Police Station, Grocery Store, Hospital, Post Office, City Hall, Cinema, Clinic, Veterinary Clinic and Bookstore (premise sources: 23, 12, 27, 2, 7, 16, 18, 8, 21 and 13). So Bookstore has a greater x-coordinate than Bakery.
Bakery never qualifies for the queried relation.
The x-coordinates increase strictly along Park, Gym, Train Station, Bank, Gallery and Bookstore (premise sources: 1, 28, 17, 5 and 19). So Bookstore has a greater x-coordinate than Park.
Park never qualifies for the queried relation.
The x-coordinates increase strictly along Supermarket, Pharmacy, Museum, Stadium, Library, Bank, Gallery and Bookstore (premise sources: 25, 20, 11, 24, 14, 5 and 19). So Bookstore has a greater x-coordinate than Supermarket.
Supermarket never qualifies for the queried relation.
The x-coordinates increase strictly along Courthouse, Stadium, Library, Bank, Gallery and Bookstore (premise sources: 3, 24, 14, 5 and 19). So Bookstore has a greater x-coordinate than Courthouse.
Courthouse never qualifies for the queried relation.
Bookstore has a greater x-coordinate than High School (premise sources: 15).
High School never qualifies for the queried relation.
The x-coordinates increase strictly along Florist, Museum, Stadium, Library, Bank, Gallery and Bookstore (premise sources: 10, 11, 24, 14, 5 and 19). So Bookstore has a greater x-coordinate than Florist.
Florist never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Bakery, Park, Supermarket, Courthouse, High School and Florist are west of Bookstore. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: D

### Symbolic

[p23, p12, p27, p2, p7, p16, p18, p8, p21, p13] x(Bakery) < x(Gas Station) < x(Police Station) < x(Grocery Store) < x(Hospital) < x(Post Office) < x(City Hall) < x(Cinema) < x(Clinic) < x(Veterinary Clinic) < x(Bookstore) => x(Bakery) < x(Bookstore).
NOT E(Bakery,Bookstore) in every valid arrangement.
[p1, p28, p17, p5, p19] x(Park) < x(Gym) < x(Train Station) < x(Bank) < x(Gallery) < x(Bookstore) => x(Park) < x(Bookstore).
NOT E(Park,Bookstore) in every valid arrangement.
[p25, p20, p11, p24, p14, p5, p19] x(Supermarket) < x(Pharmacy) < x(Museum) < x(Stadium) < x(Library) < x(Bank) < x(Gallery) < x(Bookstore) => x(Supermarket) < x(Bookstore).
NOT E(Supermarket,Bookstore) in every valid arrangement.
[p3, p24, p14, p5, p19] x(Courthouse) < x(Stadium) < x(Library) < x(Bank) < x(Gallery) < x(Bookstore) => x(Courthouse) < x(Bookstore).
NOT E(Courthouse,Bookstore) in every valid arrangement.
[p15] x(High School) < x(Bookstore).
NOT E(High School,Bookstore) in every valid arrangement.
[p10, p11, p24, p14, p5, p19] x(Florist) < x(Museum) < x(Stadium) < x(Library) < x(Bank) < x(Gallery) < x(Bookstore) => x(Florist) < x(Bookstore).
NOT E(Florist,Bookstore) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

Case 1 (counterexample satisfying all premises): W(Bakery,Bookstore) & W(Park,Bookstore) & W(Supermarket,Bookstore) & W(Courthouse,Bookstore) & W(High School,Bookstore) & W(Florist,Bookstore). Qualifying objects = {}. count=0.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: D

## production-b2904-s345-which-m0-unique-entailed

Categories: sole-entailed-with-contingent.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Supermarket, Grocery Store, Hotel, Library, Veterinary Clinic, Museum, Fire Station
Query: which North Supermarket
Eligible: Grocery Store, Hotel, Library, Veterinary Clinic, Museum, Fire Station
Consider a map with multiple objects:
Veterinary Clinic is to the Northeast of Fire Station.
Library is to the West of Veterinary Clinic.
Grocery Store is to the Northwest of Museum.
Library is to the North of Supermarket.
Library is to the Northeast of Fire Station.
Grocery Store is to the West of Hotel.
Question: Which single eligible candidate is individually guaranteed to be north of Supermarket?
A. Veterinary Clinic
B. Grocery Store
C. Hotel
D. Library
E. Cannot be determined
F. None of the above

### Natural

Supermarket and Library share the same x-coordinate (premise sources: 4).
Library has a greater y-coordinate than Supermarket (premise sources: 4).
Therefore, Library is north of Supermarket in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Grocery Store, Hotel, Museum and Fire Station are northwest of Supermarket. Library is north of Supermarket. Veterinary Clinic is northeast of Supermarket. Only Library qualifies in this arrangement.

The claim that Grocery Store is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Hotel is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Veterinary Clinic is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Museum is north of Supermarket is false in Case 1, so it is not entailed.

The claim that Fire Station is north of Supermarket is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Library}.

Answer: D

### Symbolic

[p4] x(Supermarket) = x(Library).
[p4] y(Supermarket) < y(Library).
N(Library,Supermarket) in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Grocery Store,Supermarket) & NW(Hotel,Supermarket) & NW(Museum,Supermarket) & NW(Fire Station,Supermarket). N(Library,Supermarket). NE(Veterinary Clinic,Supermarket). Qualifying objects = {Library}.

The claim that N(Grocery Store,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Hotel,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Veterinary Clinic,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Museum,Supermarket) is false in Case 1, so it is not entailed.

The claim that N(Fire Station,Supermarket) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Library}.

Answer: D
