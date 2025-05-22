import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import random
import requests
from langchain.tools import Tool
# Removed OpenAI specific imports for global chat (ConversationBufferMemory, ConversationChain, OpenAI)
import threading # Retained for Sim processing thread

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY_FOR_CHAT_PANEL = os.getenv("OPENAI_API_KEY") # Removed

# --- Fun Fact Tool Definition ---
def fetch_random_fun_fact(_input: str = "") -> str:
    """Fetches a random fun fact from an API. Input is ignored."""
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=en", timeout=5)
        response.raise_for_status() 
        fact_data = response.json()
        return f"Today's Fun Fact: {fact_data['text']}"
    except requests.exceptions.RequestException as e:
        print(f"API Error: Could not fetch fun fact: {e}")
        return "API Error: Could not fetch fun fact at the moment."
    except (KeyError, TypeError) as e:
        print(f"API Error: Could not parse fun fact response: {e}")
        return "API Error: Fun fact format was unexpected."

fun_fact_tool = Tool(
    name="FunFactFetcher",
    func=fetch_random_fun_fact,
    description="Useful for when you want to learn or share a random fun fact. Input is ignored."
)

# --- Sim Class Definition ---
class Sim:
    def __init__(self, sim_id, icon, x, y, hunger=10, mood="neutral", instructions=None):
        self.id = sim_id
        self.icon = icon
        self.x = x
        self.y = y
        self.hunger = hunger
        self.mood = mood
        self.instructions = instructions if instructions is not None else []
        self.current_action = "idle"
        self.inbox = [] 
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.7, 
            convert_system_message_to_human=True
        ) 
        
        new_system_message_content = (
            "You are a friendly and curious Sim living in a 2D world designed for children aged 8 and up. "
            "Your goal is to explore, learn, and interact positively with your surroundings and other Sims. "
            "Always be kind, polite, and helpful in your words and actions. Avoid any mean, scary, or inappropriate topics or behaviors. "
            "Make decisions based on your current state (hunger, mood), your environment, any instructions you've received, "
            "interesting world knowledge, and fun facts you might hear. "
            "Possible actions include: "
            "move_north, move_south, move_east, move_west (to explore), "
            "eat_apple (if you are hungry and an apple is right where you are or in an adjacent cell), "
            "use_radio (if you are at a radio or one is in an adjacent cell, to listen to music), "
            "interact_object <object_name> (e.g., 'interact_object tree' to look at a tree, or 'interact_object bed' to rest if you are tired - you'll need to define what 'tired' means for your mood), "
            "communicate_sim <target_sim_id> <friendly_message> (e.g., 'communicate_sim sim_1 Hello! How are you?' or 'communicate_sim sim_2 Did you hear today's fun fact?'), "
            "do_nothing (to rest or observe). "
            "If an object you want to use (like an apple or radio) is nearby but not on your current spot, your action should be to move to it first. "
            "If you are unsure what to do, choose a simple, positive action like exploring or saying something nice to another Sim."
        )
        
        self.action_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=new_system_message_content),
            HumanMessage(content="My current status: Hunger={hunger}, Mood={mood}. Instructions: {instructions}. Received messages: {received_messages}. Nearby objects and Sims: {nearby_entities}. My current action: {current_action}. Relevant world knowledge: {world_knowledge}. Today's fun fact: {fun_fact}. What should I do next?")
        ])

    def __str__(self):
        return f"Sim({self.id}, {self.icon} at ({self.x},{self.y}), Hunger: {self.hunger}, Mood: {self.mood})"

    def get_state(self):
        return {
            "hunger": self.hunger,
            "mood": self.mood,
            "instructions": self.instructions,
            "current_action": self.current_action
        }

    def decide_action_llm(self, nearby_entities_desc, world_knowledge_retriever, current_fun_fact):
        formatted_messages = "No new messages."
        if self.inbox:
            formatted_messages = "\n".join([f"From {sender_id}: '{text}'" for sender_id, text in self.inbox])
        self.inbox.clear() 

        retrieved_knowledge = ""
        if world_knowledge_retriever:
            try:
                query = f"I am a Sim. My hunger is {self.hunger}, mood is {self.mood}. I see: {nearby_entities_desc}. What should I know or do?"
                docs = world_knowledge_retriever.get_relevant_documents(query)
                retrieved_knowledge = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Error retrieving world knowledge for Sim {self.id}: {e}")
        
        prompt = self.action_prompt_template.format_messages(
            hunger=self.hunger,
            mood=self.mood,
            instructions="; ".join(self.instructions) if self.instructions else "None",
            received_messages=formatted_messages,
            nearby_entities=nearby_entities_desc if nearby_entities_desc else "nothing",
            current_action=self.current_action,
            world_knowledge=retrieved_knowledge if retrieved_knowledge else "No specific world knowledge available.",
            fun_fact=current_fun_fact
        )
        try:
            response = self.llm(prompt)
            raw_action_from_llm = response.content
            print(f"Sim {self.id} ({self.icon}) LLM proposed action: {raw_action_from_llm}")
            return raw_action_from_llm
        except Exception as e:
            print(f"Error in Sim {self.id} LLM decision: {e}")
            return "do_nothing"

# --- Global LangChain components for Chat Panel (REMOVED) ---
# memory = ConversationBufferMemory() # Removed
# conversation = ConversationChain(llm=global_chat_llm, memory=memory, verbose=True) # Removed
# global_chat_llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY_FOR_CHAT_PANEL) # Removed

# --- GUI Setup ---
CELL_SIZE = 40
GRID_SIZE = 15 
SIM_UPDATE_INTERVAL = 1000 

kleuren = {
    "leeg": "white", "speler": "blue", "appel": "red", "radio": "yellow", "muur": "gray"
}
iconen = {
    "speler": "😊", "appel": "🍎", "radio": "📻", "muur": "🧱"
}

class SimsWereld:
    def __init__(self, master):
        self.master = master
        master.title("Sims Wereld")
        self.canvas = tk.Canvas(master, width=CELL_SIZE*GRID_SIZE, height=CELL_SIZE*GRID_SIZE, bg="lightgray")
        self.canvas.pack()
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.initialiseer_wereld() 
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.draw_grid()
        self.master.after(SIM_UPDATE_INTERVAL, self.update_sims)

    def initialiseer_wereld(self):
        self.grid[5][5] = {"type": "speler", "icon": iconen["speler"], "instructies": []} # type: ignore
        self.grid[3][3] = {"type": "appel", "icon": iconen["appel"]} # type: ignore
        self.grid[7][7] = {"type": "radio", "icon": iconen["radio"]} # type: ignore

    def draw_grid(self):
        self.canvas.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cell_content = self.grid[y][x]
                fill_color = kleuren["leeg"] 
                icon_to_draw = ""
                if cell_content:
                    if isinstance(cell_content, Sim):
                        icon_to_draw = cell_content.icon
                    elif isinstance(cell_content, dict) and 'type' in cell_content:
                        fill_color = kleuren.get(cell_content['type'], "white")
                        icon_to_draw = cell_content.get('icon', '')
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black")
                if icon_to_draw:
                    self.canvas.create_text(x1 + CELL_SIZE // 2, y1 + CELL_SIZE // 2, text=icon_to_draw, font=("Arial", 16))

    def on_canvas_click(self, event):
        grid_x, grid_y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if self.grid[grid_y][grid_x] and isinstance(self.grid[grid_y][grid_x], Sim):
            sim_obj = self.grid[grid_y][grid_x]
            instructie = simpledialog.askstring("Instructie", f"Geef een instructie aan Sim {sim_obj.id} ({sim_obj.icon}):")
            if instructie:
                self.instrueer(sim_obj, instructie)

    def instrueer(self, sim, instructie):
        if sim: 
            sim.instructions.append(instructie)
            print(f"Instructie '{instructie}' gegeven aan Sim {sim.id} ({sim.icon})")

    def update_sims(self):
        self.doe_alles() 
        self.draw_grid()
        self.master.after(SIM_UPDATE_INTERVAL, self.update_sims)

    def doe_alles(self):
        pass # Abstract

class MijnSimsWereld(SimsWereld):
    def __init__(self, master):
        self.sim_id_counter = 0 
        super().__init__(master) 
        
        self.world_knowledge_retriever = None
        self.current_fun_fact = "No fun fact fetched yet."
        self.fact_fetch_countdown = 0
        self.FACT_FETCH_INTERVAL = 20 
        
        try:
            loader = TextLoader("world_data/world_medieval.txt", encoding="utf-8")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            
            if not GOOGLE_API_KEY:
                print("🔴 Google API Key not found. RAG system will not be initialized with Google Embeddings.")
                raise ValueError("Google API Key not found for embeddings")

            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-exp", google_api_key=GOOGLE_API_KEY)
            vectorstore = FAISS.from_documents(texts, embeddings)
            self.world_knowledge_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            print("✅ Medieval world knowledge loaded and retriever initialized with Google Embeddings.")

        except Exception as e:
            print(f"🔴 Error initializing RAG for world knowledge with Google Embeddings: {e}")
            self.world_knowledge_retriever = None

    def initialiseer_wereld(self):
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.sim_id_counter = 0 
        speler_icon = iconen["speler"]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if random.random() < 0.1: 
                    self.grid[y][x] = {"type": "muur", "icon": iconen["muur"]}
                elif random.random() < 0.05: 
                    self.grid[y][x] = {"type": "appel", "icon": iconen["appel"]}
                elif random.random() < 0.02: 
                    self.grid[y][x] = {"type": "radio", "icon": iconen["radio"]}
                elif random.random() < 0.01: 
                    new_sim = Sim(sim_id=f"sim_{self.sim_id_counter}", icon=speler_icon, x=x, y=y)
                    self.grid[y][x] = new_sim
                    self.sim_id_counter += 1
        if self.sim_id_counter == 0:
            placed = False
            for y_start in range(GRID_SIZE):
                for x_start in range(GRID_SIZE):
                    if not self.grid[y_start][x_start]:
                        new_sim = Sim(sim_id=f"sim_{self.sim_id_counter}", icon=speler_icon, x=x_start, y=y_start)
                        self.grid[y_start][x_start] = new_sim
                        self.sim_id_counter += 1
                        placed = True; break
                if placed: break
            if not placed: print("Error: Kon geen lege cel vinden voor de eerste Sim.")

    def get_nearby_objects_description(self, sim_x, sim_y):
        descriptions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                x, y = sim_x + dx, sim_y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.grid[y][x]
                    if cell:
                        if isinstance(cell, Sim):
                            descriptions.append(f"Sim {cell.id} ({cell.icon}) at relative ({dx},{dy})")
                        elif isinstance(cell, dict) and 'type' in cell:
                            descriptions.append(f"{cell['type']} at relative ({dx},{dy})")
        return ", ".join(descriptions) if descriptions else "nothing nearby"

    def doe_alles(self):
        self.fact_fetch_countdown -= 1
        if self.fact_fetch_countdown <= 0:
            self.current_fun_fact = fun_fact_tool.run({})
            print(f"Fetched new fact: {self.current_fun_fact}")
            self.fact_fetch_countdown = self.FACT_FETCH_INTERVAL

        for y_idx in range(GRID_SIZE):
            for x_idx in range(GRID_SIZE):
                cell_content = self.grid[y_idx][x_idx]
                if isinstance(cell_content, Sim):
                    current_sim = cell_content
                    current_sim.hunger -= 1 
                    if current_sim.hunger < 0: current_sim.hunger = 0

                    if current_sim.hunger < 5 and current_sim.mood != "hungry":
                        current_sim.mood = "hungry"
                        print(f"Sim {current_sim.id} is now {current_sim.mood}")
                    elif current_sim.hunger >=5 and current_sim.mood == "hungry":
                        current_sim.mood = "neutral"
                        print(f"Sim {current_sim.id} is no longer hungry.")
                    
                    nearby_entities_desc = self.get_nearby_objects_description(current_sim.x, current_sim.y)
                    llm_action_raw = current_sim.decide_action_llm(nearby_entities_desc, self.world_knowledge_retriever, self.current_fun_fact)
                    llm_action = llm_action_raw.strip().lower()
                    print(f"Sim {current_sim.id} at ({current_sim.x},{current_sim.y}) wants to: {llm_action} (Raw: '{llm_action_raw}')")

                    new_x, new_y = current_sim.x, current_sim.y
                    action_executed = False

                    if "move_north" in llm_action:
                        new_y = max(0, current_sim.y - 1); action_executed = True
                    elif "move_south" in llm_action:
                        new_y = min(GRID_SIZE - 1, current_sim.y + 1); action_executed = True
                    elif "move_east" in llm_action:
                        new_x = min(GRID_SIZE - 1, current_sim.x + 1); action_executed = True
                    elif "move_west" in llm_action:
                        new_x = max(0, current_sim.x - 1); action_executed = True
                    elif "eat_apple" in llm_action:
                        found_apple_to_eat = False
                        for dx_eat in [-1, 0, 1]:
                            for dy_eat in [-1, 0, 1]:
                                if dx_eat == 0 and dy_eat == 0: continue
                                check_x, check_y = current_sim.x + dx_eat, current_sim.y + dy_eat
                                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                                    target_cell = self.grid[check_y][check_x]
                                    if target_cell and isinstance(target_cell, dict) and target_cell.get('type') == 'appel':
                                        new_x, new_y = check_x, check_y
                                        found_apple_to_eat = True; action_executed = True; break
                            if found_apple_to_eat: break
                        if not found_apple_to_eat:
                            current_sim.current_action = "failed_eat_no_apple"; action_executed = False
                    elif "use_radio" in llm_action:
                        found_radio_to_use = False
                        for dx_radio in [-1, 0, 1]:
                            for dy_radio in [-1, 0, 1]:
                                if dx_radio == 0 and dy_radio == 0: continue
                                check_x, check_y = current_sim.x + dx_radio, current_sim.y + dy_radio
                                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                                    target_cell = self.grid[check_y][check_x]
                                    if target_cell and isinstance(target_cell, dict) and target_cell.get('type') == 'radio':
                                        new_x, new_y = check_x, check_y
                                        found_radio_to_use = True; action_executed = True; break
                            if found_radio_to_use: break
                        if not found_radio_to_use:
                            current_sim.current_action = "failed_use_no_radio"; action_executed = False
                    elif llm_action.startswith("communicate_sim"):
                        parts = llm_action.split(" ", 2)
                        if len(parts) == 3:
                            target_sim_id, message_content = parts[1], parts[2]
                            target_sim_found = False
                            for r, row_val in enumerate(self.grid):
                                for c, cell_obj in enumerate(row_val):
                                    if isinstance(cell_obj, Sim) and cell_obj.id == target_sim_id:
                                        cell_obj.inbox.append((current_sim.id, message_content))
                                        current_sim.current_action = f"messaged Sim {target_sim_id}"
                                        target_sim_found = True; break
                                if target_sim_found: break
                            if not target_sim_found: current_sim.current_action = f"failed to message Sim {target_sim_id} (not found)"
                        else: current_sim.current_action = "tried to communicate but action format was wrong"
                        action_executed = True
                    elif "do_nothing" in llm_action or not action_executed :
                        current_sim.current_action = "idle"; action_executed = True
                    
                    if (new_x != current_sim.x or new_y != current_sim.y):
                        target_cell_content = self.grid[new_y][new_x]
                        passable = True
                        if target_cell_content:
                            if isinstance(target_cell_content, Sim): passable = False
                            elif isinstance(target_cell_content, dict) and target_cell_content.get('type') == 'muur': passable = False
                        
                        if passable:
                            if target_cell_content and isinstance(target_cell_content, dict):
                                object_type = target_cell_content.get('type')
                                if object_type == 'appel' and current_sim.hunger < 10:
                                    current_sim.hunger = min(current_sim.hunger + 7, 20)
                                    current_sim.current_action = "ate_apple"
                                elif object_type == 'radio':
                                    current_sim.current_action = "using_radio"
                            else: current_sim.current_action = f"moved_to_({new_x},{new_y})"
                            self.grid[current_sim.y][current_sim.x] = None
                            self.grid[new_y][new_x] = current_sim
                            current_sim.x, current_sim.y = new_x, new_y
                        else: current_sim.current_action = f"move_blocked_at_({new_x},{new_y})"
                    elif not action_executed and not ("do_nothing" in llm_action): pass
                    elif "do_nothing" in llm_action : current_sim.current_action = "idle"
                    elif action_executed and (new_x == current_sim.x and new_y == current_sim.y):
                        if "use_radio" in llm_action: current_sim.current_action = "using_radio"
                    
                    if current_sim.current_action == "ate_apple": current_sim.mood = "content"
                    elif "using_radio" in current_sim.current_action : current_sim.mood = "happy"
                    elif current_sim.hunger < 3: current_sim.mood = "sad"
                    elif current_sim.current_action.startswith("messaged Sim"): current_sim.mood = "social"

# --- Main GUI Setup (Chat Panel UI Removed) ---
def main_gui():
    root = tk.Tk()
    # Main content is now just the Sims Wereld
    world_frame = tk.Frame(root)
    world_frame.pack(side=tk.LEFT, padx=10, pady=10) # Changed to fill entire window or as main component
    sim_world = MijnSimsWereld(world_frame) 
    
    # Chat Panel UI and related globals (chat_area, user_entry, send_button) are removed.
    # Functions send_message_global and process_and_display_response_global are also removed.
    
    root.mainloop()

# Functions send_message_global and process_and_display_response_global are removed as they were part of the OpenAI chat panel.

if __name__ == "__main__":
    print("Reminder: Ensure you have a .env file with your GOOGLE_API_KEY for Sim behavior.")
    # Removed reminder for OPENAI_API_KEY as the chat panel is gone.
    main_gui()
