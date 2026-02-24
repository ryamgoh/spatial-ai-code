import json
import random
import os

class SpatialDataGenerator:
    def __init__(self):
        # Define movement vectors
        self.directions = {
            "North": (0, 10), "South": (0, -10),
            "East": (10, 0), "West": (-10, 0),
            "Northeast": (10, 10), "Northwest": (-10, 10),
            "Southeast": (10, -10), "Southwest": (-10, -10)
        }
        
        # Expanded place name library (50 items)
        self.place_names = [
            # Original set
            "Police Supply Store", "Narwhal's Novelties", "Coral Crafts", 
            "Planetarium Prints", "Oz Oddities", "Ice Queen Ice Cream",
            "Burger Barn", "Taco Town", "Gadget Grove", "Book Bunker",
            "Coffee Corner", "Donut Den", "Flower Fort", "Pizza Palace",
            "Sushi Spot", "Ramen Realm", "Tea Terrace", "Video Village",
            
            # Expanded set
            "Arcade Alley", "Bakery Bay", "Bank Block", "Barber Barn",
            "Candy Castle", "Cinema City", "Circus Corner", "Diner Drive",
            "Egg Emporium", "Farm Field", "Fruit Factory", "Gym Garden",
            "Hotel Hill", "Juice Junction", "Kite Kingdom", "Library Lane",
            "Market Maze", "Museum Mound", "Music Mill", "Noodle Nook",
            "Office Oasis", "Park Plaza", "Pet Palace", "Quiz Quarter",
            "Robot Room", "School Square", "Shoe Shop", "Stadium Stand",
            "Toy Tower", "University Unit", "Villa Valley", "Water World",
            "Yogurt Yard", "Zoo Zone"
        ]

    def get_logic_string(self, direction, ref_x, ref_y):
        """Generate mathematical inequality logic based on direction."""
        logic_map = {
            "North": f"x = {ref_x}, y > {ref_y}",
            "South": f"x = {ref_x}, y < {ref_y}",
            "East": f"x > {ref_x}, y = {ref_y}",
            "West": f"x < {ref_x}, y = {ref_y}",
            "Northeast": f"x > {ref_x}, y > {ref_y}",
            "Northwest": f"x < {ref_x}, y > {ref_y}",
            "Southeast": f"x > {ref_x}, y < {ref_y}",
            "Southwest": f"x < {ref_x}, y < {ref_y}"
        }
        return logic_map.get(direction, "Unknown logic")

    def get_relative_direction(self, p1, p2):
        """Calculate the direction of p2 relative to p1."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0 and dy > 0: return "North"
        if dx == 0 and dy < 0: return "South"
        if dx > 0 and dy == 0: return "East"
        if dx < 0 and dy == 0: return "West"
        if dx > 0 and dy > 0: return "Northeast"
        if dx < 0 and dy > 0: return "Northwest"
        if dx > 0 and dy < 0: return "Southeast"
        if dx < 0 and dy < 0: return "Southwest"
        return None

    def generate_scenario(self, min_steps=4, max_steps=10):
        # 1. Randomly determine the number of reasoning steps for this generation
        num_steps = random.randint(min_steps, max_steps)
        
        # Ensure there are enough place names
        if num_steps > len(self.place_names):
            raise ValueError("Not enough place names for the requested number of steps.")
            
        selected_places = random.sample(self.place_names, num_steps)
        
        # 2. Build World Coordinates
        coords = {selected_places[0]: (0, 0)}
        
        # Record generation process
        construction_steps = [] 
        description_sentences = []
        
        # Initial point
        construction_steps.append({
            'type': 'origin',
            'target': selected_places[0],
            'coord': (0, 0)
        })

        # Generate subsequent points step-by-step
        for i in range(1, num_steps):
            current_place = selected_places[i]
            prev_place = selected_places[i-1]
            prev_coord = coords[prev_place]
            
            # Random walk
            move_dir_name = random.choice(list(self.directions.keys()))
            move_vec = self.directions[move_dir_name]
            
            new_x = prev_coord[0] + move_vec[0]
            new_y = prev_coord[1] + move_vec[1]
            coords[current_place] = (new_x, new_y)
            
            # Record Primary Relation (Chain)
            relations = []
            relations.append({
                'ref_entity': prev_place,
                'ref_coord': prev_coord,
                'direction': move_dir_name
            })
            description_sentences.append(f"{current_place} is to the {move_dir_name} of {prev_place}.")
            
            # --- Dynamic Cross-Validation (Randomized Intersection) ---
            if i > 1:
                # Get all potential ancestor nodes (excluding the immediate predecessor i-1)
                potential_ancestors = selected_places[:i-1]
                
                # Randomly decide how many extra clues to add
                # Logic: Can be 0 or more. Cap at 3 to maintain readability.
                max_extras = min(3, len(potential_ancestors))
                num_extras = random.randint(0, max_extras)
                
                if num_extras > 0:
                    chosen_ancestors = random.sample(potential_ancestors, num_extras)
                    
                    for ancestor in chosen_ancestors:
                        ancestor_coord = coords[ancestor]
                        rel_dir = self.get_relative_direction(ancestor_coord, (new_x, new_y))
                        
                        # Only add description if a standard 8-direction is formed between points
                        if rel_dir:
                            relations.append({
                                'ref_entity': ancestor,
                                'ref_coord': ancestor_coord,
                                'direction': rel_dir
                            })
                            description_sentences.append(f"{current_place} is to the {rel_dir} of {ancestor}.")
            
            construction_steps.append({
                'type': 'locate',
                'target': current_place,
                'coord': (new_x, new_y),
                'relations': relations
            })

        # 3. Construct Question
        target_entity = selected_places[-1]
        origin_entity = selected_places[0]
        final_answer_dir = self.get_relative_direction(coords[origin_entity], coords[target_entity])
        
        # If start and end overlap or have a tricky angle (non-8-direction), regenerate
        if not final_answer_dir:
            return self.generate_scenario(min_steps, max_steps)

        question = f"In which direction is {target_entity} relative to {origin_entity}?"
        
        # 4. Generate Reasoning Text (Chain of Thought)
        reasoning_lines = []
        
        for idx, step in enumerate(construction_steps, 1):
            if step['type'] == 'origin':
                reasoning_lines.append(f"Step {idx}: Reference Point")
                reasoning_lines.append(f"- Entity: \"{step['target']}\"")
                reasoning_lines.append(f"- Assignment: Set as origin {step['coord']}.")
            else:
                reasoning_lines.append(f"Step {idx}: Locate \"{step['target']}\"")
                
                logic_constraints = []
                for r_idx, rel in enumerate(step['relations'], 1):
                    logic = self.get_logic_string(rel['direction'], rel['ref_coord'][0], rel['ref_coord'][1])
                    logic_constraints.append(logic)
                    
                    # Format: Relation [N] -> implies ...
                    prefix = f"Relation {r_idx}" if len(step['relations']) > 1 else "Relation"
                    reasoning_lines.append(
                        f"- {prefix}: {rel['direction']} of \"{rel['ref_entity']}\" {rel['ref_coord']} -> implies {logic}."
                    )
                
                # Only generate Intersection line if multiple relations exist
                if len(step['relations']) > 1:
                    reasoning_lines.append(f"- Intersection: {' AND '.join(logic_constraints)}.")
                
                reasoning_lines.append(f"- Assignment: Set to {step['coord']}.")
        
        # Final Check Step
        reasoning_lines.append(f"Step {len(construction_steps) + 1}: Final Check")
        reasoning_lines.append(f"- Target: Direction of \"{target_entity}\" {coords[target_entity]} from \"{origin_entity}\" {coords[origin_entity]}.")
        
        # Calculate change
        dx = coords[target_entity][0]
        dy = coords[target_entity][1]
        calc_str = []
        if dx > 0: calc_str.append(f"x increases (0->{dx})")
        elif dx < 0: calc_str.append(f"x decreases (0->{dx})")
        else: calc_str.append("x stays same")
        
        if dy > 0: calc_str.append(f"y increases (0->{dy})")
        elif dy < 0: calc_str.append(f"y decreases (0->{dy})")
        else: calc_str.append("y stays same")
        
        reasoning_lines.append(f"- Calculation: {', '.join(calc_str)}.")
        reasoning_lines.append(f"- Result: {final_answer_dir}.")
        
        full_reasoning = "\n\n".join(reasoning_lines)

        # 5. Construct Options
        options = ["Northeast", "Northwest", "Southwest", "Southeast"]
        if final_answer_dir not in options:
            options[0] = final_answer_dir
        
        option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        formatted_options = "\n\n".join([f"{option_map[i]}. {opt}" for i, opt in enumerate(options)])
        
        correct_option_char = ""
        for i, opt in enumerate(options):
            if opt == final_answer_dir:
                correct_option_char = option_map[i]
                break

        # 6. Final JSON Assembly
        random.shuffle(description_sentences)
        map_description = " ".join(description_sentences)
        
        user_prompt = f"""Consider a map with multiple objects:

{map_description}

Please answer the following multiple-choice question based on the provided information. {question} Available options:

{formatted_options}."""

        return {
            "system": "You are a spatial reasoning expert. Your task is to:\n\n1. Analyze the map description.\n2. Assign integer 2D coordinates (x, y) to each location. Use a scale of 10 units per step to avoid conflicts.\n3. Answer the multiple-choice question.\n\nCoordinate Rules:\n\n- North: y increases (+y)\n- South: y decreases (-y)\n- East: x increases (+x)\n- West: x decreases (-x)\n\nOutput Format:\n\nReturn a valid JSON object with keys: \"reasoning\" (string), \"coordinates\" (object), and \"answer\" (string).",
            "user": user_prompt,
            "assistant": {
                "reasoning": full_reasoning,
                "coordinates": coords,
                "answer": correct_option_char
            }
        }

# --- Main Execution ---

if __name__ == "__main__":
    generator = SpatialDataGenerator()
    dataset = []
    
    # Generate 2000 samples for the initial fine-tuning phase
    print("Generating data...")
    for i in range(500):
        # Randomly vary complexity between 4 and 10 steps
        data = generator.generate_scenario(min_steps=4, max_steps=10)
        dataset.append(data)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} samples...")

    # Define output file path
    output_file = "../reason_train.json"

    # Write data to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(dataset)} items to {output_file}")
    except FileNotFoundError:
        print(f"Error: Path {output_file} not found. Ensure the parent directory exists.")
    except Exception as e:
        print(f"Error saving file: {e}")