import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import random
import requests
from langchain.tools import Tool
import threading
import warnings
import time # For potential diagnostic sleep
from rag import process_urls
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.schema import Document

from langchain.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Fun Fact Tool Definition ---
def fetch_random_fun_fact(_input: str = "") -> str:
    """Fetches a random fun fact from an API. Input is ignored."""
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=en", timeout=5)
        response.raise_for_status()
        fact_data = response.json()
        return f"Today's Fun Fact: {fact_data['text']}"
    except requests.exceptions.RequestException as e:
        return "API Error: Could not fetch fun fact at the moment."
    except (KeyError, TypeError) as e:
        return "API Error: Fun fact format was unexpected."

# Tool object for fun fact fetching, can be used by LLMs
fun_fact_tool = Tool(
    name="FunFactFetcher",
    func=fetch_random_fun_fact,
    description="Useful for when you want to learn or share a random fun fact. Input is ignored."
)

# --- Sim Class Definition ---
class Sim:
    def __init__(self, sim_id, icon, x, y, hunger=30, mood="neutral", instructions=None):
        # Sim state initialization
        self.id = sim_id
        self.icon = icon
        self.x = x
        self.y = y
        self.hunger = hunger
        self.mood = mood
        self.instructions = instructions if instructions is not None else []
        self.current_action = "idle"
        self.inbox = []
        self.is_thinking = False
        self.llm_action_result = None
        self._llm_params_cache = {}
        self.money = 0 

        # LLM setup for Sim's decision making
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        )

        # System prompt for the Sim's LLM
        new_system_message_content = (
            "You are a friendly and curious Sim living in a 2D world. "
            "Your goal is to explore, learn, and interact positively. Always be kind and safe. "
            "Make decisions based on your state (hunger, mood), environment, instructions, messages, world knowledge, and fun facts. "
            "Respond with ONLY ONE of the following action commands: "
            "move_north, move_south, move_east, move_west (to explore), "
            "eat_apple (if hungry and an apple is at your current location or in an adjacent cell, you can use it without moving), "
            "use_trumpet (if a trumpet is at your current location or in an adjacent cell—if adjacent, you can use it without moving), "
            "use_farm (if a farm is at your current location or in an adjacent cell—if adjacent, you can use it without moving), "
            "interact_object <object_name> (e.g., 'interact_object tree', 'interact_object bed' or 'interact_object farm', you can use it without moving), "
            "communicate_sim <target_sim_id> <friendly_message> (e.g., 'communicate_sim sim_1 Hello!'), "
            "do_nothing (to rest or observe). "
            "If an object is nearby but not on your spot, your action should be to move to it first (e.g. 'move_east' if apple is east). "
            "If unsure, choose a simple positive action like exploring."
            "Try to avoid 'do_nothing' and prefer exploring, eating, communicating, or interacting with objects."
            "The Sim environment plays in the medieval times, so your actions and interactions should reflect that setting."
        )

        # Prompt template for LLM input
        self.action_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=new_system_message_content),
            HumanMessagePromptTemplate.from_template(
                (
                    "My current status: Hunger={hunger}, Mood={mood}.\n"
                    "My last action: {current_action}.\n"
                    "Instructions: {instructions}.\n"
                    "Received messages: {received_messages}.\n"
                    "Nearby objects and Sims: {nearby_entities}.\n"
                    "Relevant world knowledge: {world_knowledge}.\n"
                    "Today's fun fact: {fun_fact}.\n"
                    "What should I do next? Respond with ONLY ONE action command."
                )
            )
        ])

    def __str__(self):
        # String representation for debugging/logging
        return f"Sim({self.id}, {self.icon} at ({self.x},{self.y}), H:{self.hunger}, M:{self.mood}, Act:{self.current_action}, Think:{self.is_thinking}, Res:'{self.llm_action_result}')"

    def prepare_llm_decision_params(self, nearby_entities_desc, world_knowledge_retriever, current_fun_fact):
        # Cache parameters for LLM decision making
        self._llm_params_cache['nearby_entities_desc'] = nearby_entities_desc
        self._llm_params_cache['world_knowledge_retriever'] = world_knowledge_retriever
        self._llm_params_cache['current_fun_fact'] = current_fun_fact

    def _run_llm_decision_in_thread(self):
        # print(f"DEBUG_THREAD_START: Sim {self.id} _run_llm_decision_in_thread.")
        # Runs the LLM decision logic in a background thread
        nearby_entities_desc = self._llm_params_cache.get('nearby_entities_desc')
        world_knowledge_retriever = self._llm_params_cache.get('world_knowledge_retriever')
        current_fun_fact = self._llm_params_cache.get('current_fun_fact')

        formatted_messages = "No new messages."
        if self.inbox:
            formatted_messages = "\n".join([f"From {sender_id}: '{text}'" for sender_id, text in self.inbox])

        retrieved_knowledge = ""
        if world_knowledge_retriever:
            try:
                # Query the retriever for relevant world knowledge
                query = f"Sim state: hunger {self.hunger}, mood {self.mood}. Nearby: {nearby_entities_desc}. What's relevant?"
                docs = world_knowledge_retriever.get_relevant_documents(query)
                retrieved_knowledge = "\n".join([doc.page_content for doc in docs])
                #print(f"DEBUG_RAG_CONTEXT for Sim {self.id}:\n{retrieved_knowledge}\n{'-'*40}")

            except Exception as e:
                print(f"ERROR: Retrieving world knowledge for {self.id}: {e}")
                retrieved_knowledge = "Error retrieving knowledge."

        # Prepare prompt input for the LLM
        prompt_input = {
            "hunger": self.hunger,
            "mood": self.mood,
            "current_action": self.current_action,
            "instructions": "; ".join(self.instructions) if self.instructions else "None",
            "received_messages": formatted_messages,
            "nearby_entities": nearby_entities_desc if nearby_entities_desc else "nothing",
            "world_knowledge": retrieved_knowledge if retrieved_knowledge else "No specific world knowledge.",
            "fun_fact": current_fun_fact
        }
        
        try:
            # Format and send prompt to LLM, get response
            formatted_prompt_messages = self.action_prompt_template.format_messages(**prompt_input)
            #print(f"DEBUG_LLM_PROMPT: Sim {self.id} (Autonomous) PROMPT for LLM:\n{'-'*20}\n{formatted_prompt_messages}\n{'-'*20}")
            response = self.llm.invoke(formatted_prompt_messages)
            print(f"DEBUG_LLM_RESPONSE: Sim {self.id} (Autonomous) RAW LLM RESPONSE: '{response.content}'")
            self.llm_action_result = response.content.strip()
            print(f"DEBUG_THREAD_RESULT: Sim {self.id} _run_llm_decision_in_thread - llm_action_result SET TO: '{self.llm_action_result}'")
            if self.inbox: self.inbox.clear() # Clear inbox after processing for this turn.
        except Exception as e:
            print(f"ERROR: LLM decision error for Sim {self.id} (Autonomous): {e}")
            self.llm_action_result = "do_nothing"
            print(f"DEBUG_THREAD_RESULT: Sim {self.id} _run_llm_decision_in_thread - llm_action_result SET TO 'do_nothing' due to error.")
        finally:
            self.is_thinking = False
            # print(f"DEBUG_THREAD_END: Sim {self.id} _run_llm_decision_in_thread, is_thinking now: {self.is_thinking}.")

    def start_llm_decision_thread(self):
        # Start the LLM decision thread if not already thinking
        if not self.is_thinking:
            self.llm_action_result = None # Clear previous result first
            self.is_thinking = True       # Then indicate thinking
            # print(f"DEBUG_SIM_STATE: Sim {self.id} starting LLM decision thread. is_thinking set to True.")
            decision_thread = threading.Thread(target=self._run_llm_decision_in_thread)
            decision_thread.daemon = True
            decision_thread.start()
            return True
        # else:
            # print(f"DEBUG_SIM_STATE: Sim {self.id} tried to start LLM decision thread but was already thinking.")
        return False

# --- GUI Setup ---
CELL_SIZE = 40
GRID_SIZE = 15
SIM_UPDATE_INTERVAL = 4000 # ms between Sim updates

# Color and icon mapping for grid objects
kleuren = {
    "leeg": "white", "speler": "lightblue", "appel": "red", "trumpet": "yellow", "muur": "gray", "farm": "lightyellow"
}
iconen = {
    "speler": "😊", "appel": "🍎", "trumpet": "🎺", "muur": "🧱", "thinking": "🤔", "farm": "🌾"
}

class SimsWereld:
    def __init__(self, master):
        # Set up the grid and canvas for the world
        self.master = master
        self.canvas = tk.Canvas(master, width=CELL_SIZE*GRID_SIZE, height=CELL_SIZE*GRID_SIZE, bg="lightgreen")
        self.canvas.pack()
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def draw_grid(self):
        # Draw the grid and all objects/Sims on the canvas
        self.canvas.delete("all")
        for y_draw in range(GRID_SIZE):
            for x_draw in range(GRID_SIZE):
                x1, y1 = x_draw * CELL_SIZE, y_draw * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cell_content = self.grid[y_draw][x_draw]
                fill_color = kleuren["leeg"]
                icon_to_draw = ""
                text_color = "black"

                if isinstance(cell_content, Sim):
                    sim_obj = cell_content
                    fill_color = kleuren["speler"]
                    icon_to_draw = sim_obj.icon
                    if sim_obj.is_thinking:
                        icon_to_draw = iconen["thinking"]
                elif isinstance(cell_content, dict) and 'type' in cell_content:
                    fill_color = kleuren.get(cell_content['type'], "white")
                    icon_to_draw = cell_content.get('icon', '')

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="darkgray")
                if icon_to_draw:
                    self.canvas.create_text(x1 + CELL_SIZE // 2, y1 + CELL_SIZE // 2, text=icon_to_draw, font=("Arial", 16), fill=text_color)
        
        # Draw Sim info (ID and hunger) on top of Sim cells
        for y_draw in range(GRID_SIZE):
            for x_draw in range(GRID_SIZE):
                cell_content = self.grid[y_draw][x_draw]
                if isinstance(cell_content, Sim):
                    sim_obj = cell_content
                    self.canvas.create_text(
                        x_draw * CELL_SIZE + CELL_SIZE // 2,
                        y_draw * CELL_SIZE + CELL_SIZE - 8,
                        text=f"{sim_obj.id.split('_')[-1]}|H{sim_obj.hunger}",
                        font=("Arial", 7),
                        fill="black"
                    )

    def on_canvas_click(self, event):
        # Handle clicks on the grid: allow user to instruct Sims
        grid_x, grid_y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            cell_content = self.grid[grid_y][grid_x]
            if isinstance(cell_content, Sim):
                sim_obj = cell_content
                instructie = simpledialog.askstring("Instructie", f"Geef een instructie aan Sim {sim_obj.id} ({sim_obj.icon}):\n(e.g., move_north, eat_apple, say hello to sim_0)")
                if instructie:
                    self.instrueer(sim_obj, instructie)
            else:
                print(f"INFO: Clicked empty cell or object at ({grid_x}, {grid_y})")

    def instrueer(self, sim, instructie):
        # Add a user instruction to a Sim
        if sim:
            sim.instructions.append(instructie)
            print(f"INFO: Instructed Sim {sim.id}: {instructie}")

    def update_sims(self):
        # Main update loop: initiate LLM decisions, process actions, redraw grid
        print("\n--- TICK START ---")
        self.initiate_sim_llm_decisions()
        self.process_pending_sim_actions()
        self.draw_grid()
        print("--- TICK END ---\n")
        self.master.after(SIM_UPDATE_INTERVAL, self.update_sims)

def move_sim_on_grid(world: 'MijnSimsWereld', sim: Sim, dx: int, dy: int) -> bool:
    # Attempt to move a Sim by (dx, dy) if the target cell is empty
    new_x = sim.x + dx
    new_y = sim.y + dy
    # print(f"DEBUG_MOVE: move_sim_on_grid for {sim.id}: current ({sim.x},{sim.y}), trying ({new_x},{new_y}) with ({dx},{dy})")

    if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
        print(f"DEBUG_MOVE: move_sim_on_grid for {sim.id}: FAILED - Out of bounds ({new_x},{new_y})")
        return False
    
    target_cell_content = world.grid[new_y][new_x]
    # print(f"DEBUG_MOVE: move_sim_on_grid for {sim.id}: Target cell ({new_x},{new_y}) content: {type(target_cell_content)} - {str(target_cell_content)[:60]}")

    if target_cell_content is None:
        world.grid[sim.y][sim.x] = None
        sim.x = new_x
        sim.y = new_y
        world.grid[new_y][new_x] = sim
        # print(f"DEBUG_MOVE: move_sim_on_grid for {sim.id}: SUCCESS - Moved to ({new_x},{new_y})")
        return True
    else:
        print(f"DEBUG_MOVE: move_sim_on_grid for {sim.id}: FAILED - Target cell ({new_x},{new_y}) is not None, it's occupied by {type(target_cell_content)}")
        return False

class MijnSimsWereld(SimsWereld):
    def __init__(self, master):
        # Initialize world, load RAG knowledge, and set up Sims and objects
        self.sim_id_counter = 0
        self.world_knowledge_retriever = None
        self.current_fun_fact = "No fun fact fetched yet."
        self.fact_fetch_countdown = 0
        self.FACT_FETCH_INTERVAL = 20

        super().__init__(master)

        # Load medieval world knowledge using RAG pipeline
        urls = [
                "https://courses.lumenlearning.com/atd-herkimer-westerncivilization/chapter/daily-medieval-life/#:~:text=For%20peasants%2C%20daily%20medieval%20life,could%20rest%20from%20their%20labors",
                "https://www.twinkl.com/teaching-wiki/life-like-for-a-medieval-peasant",
                "https://schoolhistory.co.uk/notes/lifestyle-of-medieval-peasants/"
            ]
        combined_text = process_urls(urls, chunk_size=500, overlap=50)
        chunk_size = 500
        chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
        documents = [Document(page_content=chunk) for chunk in chunks]
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        self.world_knowledge_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("Medieval world knowledge loaded and retriever initialized locally with SentenceTransformer.")

        self.initialiseer_wereld()
        self.draw_grid() # Initial draw
        self.master.after(SIM_UPDATE_INTERVAL, self.update_sims) # Start update loop

    def initialiseer_wereld(self):
        # Fill the grid with walls, apples, trumpets, Sims, and one farm at a random empty spot
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.sim_id_counter = 0
        speler_icon = iconen["speler"]

        # Place random walls
        for _ in range(GRID_SIZE * 2): 
            wx, wy = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if self.grid[wy][wx] is None:
                 self.grid[wy][wx] = {"type": "muur", "icon": iconen["muur"]}

        # Place apples, trumpets, and Sims randomly
        for y_init in range(GRID_SIZE):
            for x_init in range(GRID_SIZE):
                if self.grid[y_init][x_init] is not None: continue

                if random.random() < 0.05:
                    self.grid[y_init][x_init] = {"type": "appel", "icon": iconen["appel"]}
                elif random.random() < 0.02:
                    self.grid[y_init][x_init] = {"type": "trumpet", "icon": iconen["trumpet"]}
                elif self.sim_id_counter < 3 and random.random() < 0.03 : # Limit Sims
                    new_sim = Sim(sim_id=f"sim_{self.sim_id_counter}", icon=speler_icon, x=x_init, y=y_init)
                    self.grid[y_init][x_init] = new_sim
                    self.sim_id_counter += 1
        # Place one farm at a random empty location
        empty_cells = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if self.grid[y][x] is None]
        if empty_cells:
            farm_x, farm_y = random.choice(empty_cells)
            self.grid[farm_y][farm_x] = {"type": "farm", "icon": iconen["farm"]}
        # Ensure at least one Sim is placed
        if self.sim_id_counter == 0:
            placed = False
            empty_spots = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if self.grid[r][c] is None:
                        empty_spots.append((c,r))
            
            if empty_spots:
                x_start, y_start = random.choice(empty_spots)
                new_sim = Sim(sim_id=f"sim_{self.sim_id_counter}", icon=speler_icon, x=x_start, y=y_start)
                self.grid[y_start][x_start] = new_sim
                self.sim_id_counter += 1
                placed = True
            if not placed: print("ERROR: Could not place initial Sim, grid might be full of walls or too small.")
        print(f"INFO: Initialized world with {self.sim_id_counter} Sims.")

    def get_nearby_objects_description(self, sim_x, sim_y):
        # Describe all objects and Sims adjacent to the given position
        descriptions = []
        for dy_offset in [-1, 0, 1]:
            for dx_offset in [-1, 0, 1]:
                if dx_offset == 0 and dy_offset == 0: continue
                check_x, check_y = sim_x + dx_offset, sim_y + dy_offset
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    cell = self.grid[check_y][check_x]
                    if cell:
                        relative_pos_parts = []
                        if dy_offset == -1: relative_pos_parts.append("N")
                        elif dy_offset == 1: relative_pos_parts.append("S")
                        if dx_offset == -1: relative_pos_parts.append("W")
                        elif dx_offset == 1: relative_pos_parts.append("E")
                        
                        relative_pos = "".join(relative_pos_parts)
                        if not relative_pos: relative_pos = f"({dx_offset},{dy_offset})"

                        if isinstance(cell, Sim):
                            descriptions.append(f"Sim {cell.id.split('_')[-1]} ({cell.icon}) at {relative_pos}")
                            print(f"DEBUG_NEARBY: Found Sim {cell.id} at {relative_pos} for Sim at ({sim_x},{sim_y})")
                        elif isinstance(cell, dict) and 'type' in cell:
                            descriptions.append(f"To your {relative_pos}: {cell['type']} (you can use it from here if it's an apple, trumpet or farm)")
                            print(f"DEBUG_NEARBY: Found {cell['type']} at {relative_pos} for Sim at ({sim_x},{sim_y})")
        return ", ".join(descriptions) if descriptions else "nothing specific nearby"

    def initiate_sim_llm_decisions(self):
        # Start LLM decision threads for all Sims that need a new action
        print("DEBUG_INITIATE_DECISIONS: Starting to initiate decisions...")
        self.fact_fetch_countdown -= 1
        if self.fact_fetch_countdown <= 0:
            self.current_fun_fact = fun_fact_tool.run("")
            self.fact_fetch_countdown = self.FACT_FETCH_INTERVAL
            print(f"INFO: Fetched new Fun Fact: {self.current_fun_fact}")

        sims_needing_decision_initiation = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = self.grid[r][c]
                if isinstance(cell, Sim):
                    # Condition: Not currently thinking AND no pending action result to process
                    if not cell.is_thinking and cell.llm_action_result is None:
                        sims_needing_decision_initiation.append(cell)
        
        if not sims_needing_decision_initiation:
            print("DEBUG_INITIATE_DECISIONS: No Sims need new decisions initiated this tick.")
        else:
            print(f"DEBUG_INITIATE_DECISIONS: Sims needing new decisions initiated: {[s.id for s in sims_needing_decision_initiation]}")

        for current_sim in sims_needing_decision_initiation:
            current_sim.hunger -= 1 # Hunger decreases every tick
            if current_sim.hunger < 0: current_sim.hunger = 0
            if current_sim.hunger < 30 and current_sim.mood != "hungry":
                current_sim.mood = "hungry"
            elif current_sim.hunger >= 50 and current_sim.mood == "hungry":
                current_sim.mood = "neutral"

            if current_sim.instructions:
                # If Sim has instructions, use those for LLM prompt
                instruction = current_sim.instructions.pop(0) 
                print(f"INFO: Sim {current_sim.id} initiating instruction-based decision: {instruction}")
                system_instruction_prompt = (
                    "You are a Sim. Follow the user's instruction. Stay positive and safe. "
                    "Respond with ONLY ONE action command from the list: move_north, move_south, move_east, move_west, "
                    "eat_apple, use_trumpet, use_farm, interact_object <object_name>, "
                    "communicate_sim <target_sim_id> <message>, do_nothing."
                )
                human_message_content = (
                    f"Instruction: {instruction}\n"
                    f"Your current status: Hunger={current_sim.hunger}, Mood={current_sim.mood}, Last Action: {current_sim.current_action}.\n"
                    f"Nearby: {self.get_nearby_objects_description(current_sim.x, current_sim.y)}.\n"
                    f"Fun fact: {self.current_fun_fact}\n"
                    "Respond with ONLY ONE action command based on the instruction."
                )
                prompt_messages = [
                    SystemMessage(content=system_instruction_prompt),
                    HumanMessage(content=human_message_content)
                ]
                def run_instruction_llm_closure(sim_to_act, messages_for_llm):
                    try:
                        #print(f"DEBUG_LLM_PROMPT: Sim {sim_to_act.id} (Instruction) PROMPT for LLM:\n{'-'*20}\n{messages_for_llm}\n{'-'*20}")
                        response = sim_to_act.llm.invoke(messages_for_llm)
                        print(f"DEBUG_LLM_RESPONSE: Sim {sim_to_act.id} (Instruction) RAW LLM RESPONSE: '{response.content}'")
                        sim_to_act.llm_action_result = response.content.strip()
                        print(f"DEBUG_THREAD_RESULT: Sim {sim_to_act.id} (Instruction) - llm_action_result SET TO: '{sim_to_act.llm_action_result}'")
                    except Exception as e:
                        print(f"ERROR: LLM decision error for Sim {sim_to_act.id} (Instruction): {e}")
                        sim_to_act.llm_action_result = "do_nothing"
                        print(f"DEBUG_THREAD_RESULT: Sim {sim_to_act.id} (Instruction) - llm_action_result SET TO 'do_nothing' due to error.")
                    finally:
                        sim_to_act.is_thinking = False # Mark thinking as done
                
                current_sim.llm_action_result = None # Ensure it's clear before this new thought
                current_sim.is_thinking = True      # Mark as starting to think
                t = threading.Thread(target=run_instruction_llm_closure, args=(current_sim, prompt_messages))
                t.daemon = True
                t.start()

            else: # Autonomous action
                # No instructions, let Sim decide based on environment
                print(f"INFO: Sim {current_sim.id} initiating autonomous decision.")
                nearby_desc = self.get_nearby_objects_description(current_sim.x, current_sim.y)
                print(f"DEBUG_INITIATE_DECISIONS: Sim {current_sim.id} nearby description: '{nearby_desc}'")
                current_sim.prepare_llm_decision_params(nearby_desc, self.world_knowledge_retriever, self.current_fun_fact)
                current_sim.start_llm_decision_thread() # This sets is_thinking=True and llm_action_result=None

        print("DEBUG_INITIATE_DECISIONS: Finished initiating decisions for this tick.")

    def parse_llm_action(self, action_text: str, sim: Sim) -> dict:
        # Parse the LLM's action output into a structured action dict
        original_raw_text = action_text 
        processed_text = action_text.lower().strip().replace(".", "").replace("'", "")
        parts = processed_text.split()
        log_prefix = f"DEBUG_PARSE: Sim {sim.id} PARSING (Raw='{original_raw_text}', Processed='{processed_text}'):"
        parsed_action = {"type": "do_nothing"} 
        if not parts:
            # print(f"{log_prefix} No parts, result: {parsed_action}")
            return parsed_action

        command = parts[0]
        
        # Map command to action type
        if command == "move_north": parsed_action = {"type": "move_north"}
        elif command == "move_south": parsed_action = {"type": "move_south"}
        elif command == "move_east": parsed_action = {"type": "move_east"}
        elif command == "move_west": parsed_action = {"type": "move_west"}
        elif command == "eat_apple": parsed_action = {"type": "eat_apple"}
        elif command == "use_trumpet": parsed_action = {"type": "use_trumpet"}
        elif command == "use_farm": parsed_action = {"type": "use_farm"}
        elif command == "do_nothing": parsed_action = {"type": "do_nothing"}
        elif command == "interact_object" and len(parts) > 1:
            parsed_action = {"type": "interact_object", "object_name": " ".join(parts[1:])}
        elif command == "communicate_sim" and len(parts) >= 3:
            target_sim_id_part = parts[1]
            if not target_sim_id_part.startswith("sim_"):
                target_sim_id_part = f"sim_{target_sim_id_part}"
            message = " ".join(parts[2:])
            parsed_action = {"type": "communicate_sim", "target_sim_id": target_sim_id_part, "message": message}
        elif "move" in command or "walk" in command or "go" in command : 
            if "north" in processed_text : parsed_action = {"type": "move_north"}
            elif "south" in processed_text : parsed_action = {"type": "move_south"}
            elif "east" in processed_text : parsed_action = {"type": "move_east"}
            elif "west" in processed_text : parsed_action = {"type": "move_west"}
            else: # Generic move without clear direction
                print(f"{log_prefix} Generic move command '{command}' without clear direction. Defaulting to do_nothing.")
        
        if parsed_action["type"] == "do_nothing" and command not in ["do_nothing", ""] and original_raw_text.strip() != "":
             print(f"{log_prefix} Command '{command}' or full text '{original_raw_text}' not fully parsed to specific action, result: {parsed_action}")
        return parsed_action

    def find_sim_by_id(self, sim_id_to_find: str) -> Sim | None:
        # Find a Sim object in the grid by its ID
        for r_idx in range(GRID_SIZE):
            for c_idx in range(GRID_SIZE):
                cell = self.grid[r_idx][c_idx]
                if isinstance(cell, Sim) and cell.id == sim_id_to_find:
                    return cell
        return None

    def process_pending_sim_actions(self):
        # Process all Sims that have completed their LLM decision and are ready to act
        print(f"DEBUG_PROCESS_ACTIONS: --- Entering process_pending_sim_actions ---")
        sims_to_process_action = []
        sim_details_for_log = [] 

        # Collect Sims ready to act
        for r_idx in range(GRID_SIZE):
            for c_idx in range(GRID_SIZE):
                cell = self.grid[r_idx][c_idx]
                if isinstance(cell, Sim):
                    sim_details_for_log.append(
                        # Using the Sim's __str__ method for a concise overview
                        f"  {str(cell)}"
                    )
                    # Condition: Not currently thinking (so LLM thread is done) AND has a result
                    if not cell.is_thinking and cell.llm_action_result is not None:
                        sims_to_process_action.append(cell)
        
        print(f"DEBUG_PROCESS_ACTIONS: All Sims states at start of processing:")
        for detail in sim_details_for_log:
            print(detail)
        
        if not sims_to_process_action:
            print("DEBUG_PROCESS_ACTIONS: No Sims found with completed decisions to process.")
            print(f"DEBUG_PROCESS_ACTIONS: --- Exiting process_pending_sim_actions (No Actions To Process) ---")
            return

        print(f"DEBUG_PROCESS_ACTIONS: Found {len(sims_to_process_action)} Sim(s) with completed decisions: {[s.id for s in sims_to_process_action]}")
        
        random.shuffle(sims_to_process_action) 

        for sim in sims_to_process_action:
            print(f"DEBUG_PROCESS_ACTIONS: Processing action for Sim {sim.id}. Current llm_action_result: '{sim.llm_action_result}'")
            if sim.llm_action_result and not sim.is_thinking:
                raw_action_text = sim.llm_action_result
                parsed_details = self.parse_llm_action(raw_action_text, sim)
                action_type = parsed_details.get("type", "do_nothing")
                
                log_message = f"INFO: Sim {sim.id} (at {sim.x},{sim.y}, H:{sim.hunger}, M:{sim.mood}): Raw='{raw_action_text}', Parsed='{action_type}'"
                
                executed_successfully = True # Assume success initially for this action
                original_x, original_y = sim.x, sim.y

                # Handle each action type
                if action_type == "move_north":
                    if not move_sim_on_grid(self, sim, 0, -1): executed_successfully = False
                elif action_type == "move_south":
                    if not move_sim_on_grid(self, sim, 0, 1): executed_successfully = False
                elif action_type == "move_east":
                    if not move_sim_on_grid(self, sim, 1, 0): executed_successfully = False
                elif action_type == "move_west":
                    if not move_sim_on_grid(self, sim, -1, 0): executed_successfully = False
                elif action_type == "eat_apple":
                    # Eat apple from adjacent cell without moving
                    ate_apple = False
                    found_apple_to_eat = False
                    best_apple_pos = None
                    for dy_check in [-1, 0, 1]: 
                        for dx_check in [-1, 0, 1]:
                            if dx_check == 0 and dy_check == 0: continue 
                            if abs(dx_check) + abs(dy_check) > 1 : continue 
                            check_x, check_y = sim.x + dx_check, sim.y + dy_check
                            if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                                cell_to_check = self.grid[check_y][check_x]
                                if isinstance(cell_to_check, dict) and cell_to_check.get("type") == "appel":
                                    best_apple_pos = (check_x, check_y)
                                    found_apple_to_eat = True
                                    break
                        if found_apple_to_eat: break
                    if found_apple_to_eat and best_apple_pos:
                        # Remove the apple from the grid (eat it)
                        self.grid[best_apple_pos[1]][best_apple_pos[0]] = None
                        sim.hunger += 50
                        if sim.hunger > 100: sim.hunger = 100
                        log_message += f" -> Ate apple at ({best_apple_pos[0]},{best_apple_pos[1]})."
                        ate_apple = True
                    else: 
                        log_message += " -> Tried to eat apple, but no adjacent apple found."
                        executed_successfully = False
                elif action_type == "communicate_sim":
                    # Send a message to another Sim
                    target_id = parsed_details.get("target_sim_id")
                    msg_text = parsed_details.get("message")
                    if target_id and msg_text:
                        target_sim = self.find_sim_by_id(target_id)
                        if target_sim:
                            if target_sim == sim: log_message += f" -> Tried to talk to self."
                            else:
                                target_sim.inbox.append((sim.id, msg_text))
                                log_message += f" -> Sent '{msg_text}' to {target_id}."
                        else:
                            log_message += f" -> Failed to find Sim {target_id}."
                            executed_successfully = False
                    else:
                        log_message += f" -> Invalid communicate_sim."
                        executed_successfully = False
                elif action_type == "interact_object":
                    # Generic interaction with an object (e.g., farm)
                    obj_name = parsed_details.get("object_name", "unknown")
                    log_message += f" -> Interacted with {obj_name}."
                elif action_type == "use_trumpet":
                    # Use trumpet: improve mood
                    sim.mood = "excited"
                    log_message += " -> Used the trumpet 🎺! Mood is now excited."
                elif action_type == "use_farm":
                    # Use farm: improve mood, reduce hunger, earn money
                    sim.mood = "happy"
                    sim.hunger = max(0, sim.hunger - 10)
                    sim.money += 5
                    log_message += f" -> Worked on the farm, mood is now {sim.mood}, hunger decreased to {sim.hunger}, money increased to {sim.money}."
                elif action_type == "do_nothing":
                    log_message += " -> Did nothing."
                else: 
                    log_message += f" -> Unknown action '{action_type}'"
                    executed_successfully = False

                if executed_successfully:
                    sim.current_action = action_type
                else:
                    sim.current_action = f"failed_{action_type}"
                
                sim.llm_action_result = None # Mark action as consumed/attempted
                print(log_message)
            else:
                print(f"DEBUG_PROCESS_ACTIONS: Sim {sim.id} was in list but llm_action_result is None or is_thinking is True. Skipping. State: {str(sim)}")

        print(f"DEBUG_PROCESS_ACTIONS: --- Exiting process_pending_sim_actions ---")

# --- Main GUI Setup ---
def main_gui():
    # Start the Tkinter GUI and Sims world
    root = tk.Tk()
    root.title("Sims World")
    world_frame = tk.Frame(root)
    world_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    sim_world = MijnSimsWereld(world_frame)
    root.mainloop()

if __name__ == "__main__":
    # Ensure world data directory and file exist
    if not os.path.exists("world_data"):
        os.makedirs("world_data")
    if not os.path.exists("world_data/world_medieval.txt"):
        with open("world_data/world_medieval.txt", "w", encoding="utf-8") as f:
            f.write("A knight should be brave.\n")
            f.write("Apples are good food for energy.\n")
            f.write("Trumpets play music and news.\n")
            f.write("Exploring new places is fun.\n")
            f.write("It is polite to greet others.\n")
    print("Reminder: Ensure you have a .env file with your GOOGLE_API_KEY for Sim behavior.")
    main_gui()
