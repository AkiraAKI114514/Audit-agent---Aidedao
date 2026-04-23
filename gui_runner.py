import customtkinter as ctk
from tkinter import filedialog
import threading
import queue
import sys
import ctypes
from pathlib import Path

# 直接导入你的底层审计逻辑
import t1 

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# 用于截获底层 print 输出的类
class RedirectText:
    def __init__(self, text_queue):
        self.text_queue = text_queue

    def write(self, string):
        self.text_queue.put(string)

    def flush(self):
        pass

class AuditGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 审计系统控制台")
        self.geometry("1100x600")
        
        self.output_queue = queue.Queue()
        self.is_running = False
        self.audit_thread = None  # 记录当前运行的线程

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左侧控制面板 =================
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_columnconfigure(1, weight=1)

        self.add_label_entry(self.left_frame, "API URL:", 0, "https://api.siliconflow.cn/v1", "api_url")
        self.add_label_entry(self.left_frame, "API Key:", 1, "", "api_key", show="*")
        self.add_label_entry(self.left_frame, "Model:", 2, "deepseek-ai/DeepSeek-V3.2", "model")
        self.add_label_entry(self.left_frame, "Tavily API Key:", 3, "", "tc_api_key", show="*")
        self.add_label_entry(self.left_frame, "目标公司:", 4, "", "company")
        # 文件选择
        self.add_file_picker(self.left_frame, "目标 PDF:", 5, "pdf_entry")
        self.add_file_picker(self.left_frame, "竞品 PDF:", 6, "peer_pdf_entry")

        # 按钮区
        self.btn_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.btn_frame.grid(row=7, column=0, columnspan=3, pady=(30, 10), padx=10, sticky="ew")
        self.btn_frame.grid_columnconfigure(0, weight=1)
        self.btn_frame.grid_columnconfigure(1, weight=1)
        
        self.run_btn = ctk.CTkButton(self.btn_frame, text="▶ 运行审计", fg_color="green", hover_color="darkgreen", command=self.start_audit)
        self.run_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        # 恢复中断按钮
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="⏹ 中断", fg_color="red", hover_color="darkred", state="disabled", command=self.stop_audit)
        self.stop_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # ================= 右侧输出区 =================
        self.output_box = ctk.CTkTextbox(self, state="disabled", font=("Consolas", 12))
        self.output_box.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        self.output_box.configure(state="normal")
        self.output_box.insert(ctk.END, 
                               """【使用说明】
-- 需填写相应的 LLM 服务 API 链接与 API 密钥以开启 LLM 服务（必填项）。
-- 搜索服务目前只可以使用 Tavily API（非必填项）。
-- 须完整填写目标公司、目标 pdf 链接和竞品 pdf 链接，可以使用“浏览...”按钮选择，也可以手动填写。
-- 如手动填写 pdf 链接，请完整输入文件绝对位置，不同文件链接由逗号隔开。
            """)
        self.output_box.see(ctk.END)
        self.output_box.configure(state="disabled")

        self.check_queue()

    def add_label_entry(self, parent, label_text, row, default_val, attr_name, show=""):
        ctk.CTkLabel(parent, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = ctk.CTkEntry(parent, show=show)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        setattr(self, attr_name, entry)

    def add_file_picker(self, parent, label_text, row, attr_name):
        ctk.CTkLabel(parent, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = ctk.CTkEntry(parent)
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        btn = ctk.CTkButton(parent, text="浏览...", width=60, command=lambda: self.browse_files(entry))
        btn.grid(row=row, column=2, padx=5, pady=5)
        setattr(self, attr_name, entry)

    def browse_files(self, entry_widget):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            entry_widget.delete(0, ctk.END)
            entry_widget.insert(0, ",".join(files))

    def write_output(self, text):
        self.output_box.configure(state="normal")
        self.output_box.insert(ctk.END, text)
        self.output_box.see(ctk.END)
        self.output_box.configure(state="disabled")

    def check_queue(self):
        while not self.output_queue.empty():
            msg = self.output_queue.get()
            if msg == "DONE_SIGNAL":
                self.reset_ui()
            else:
                self.write_output(msg)
        self.after(50, self.check_queue)

    def start_audit(self):
        if self.is_running: return
        self.is_running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal") # 激活中断按钮
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", ctk.END) # 清空上次输出
        self.output_box.configure(state="disabled")
        
        self.write_output(">>> 正在初始化多模型审计引擎...\n")
        
        # 启动后台线程执行核心逻辑，并记录下来以便随时杀掉
        self.audit_thread = threading.Thread(target=self.run_audit_backend, daemon=True)
        self.audit_thread.start()

    def stop_audit(self):
        """黑魔法：通过 ctypes 强行向运行中的线程抛出异常来停止它"""
        if self.is_running and self.audit_thread and self.audit_thread.is_alive():
            self.write_output("\n>>> 正在发送中断信号，请稍候...\n")
            
            thread_id = self.audit_thread.ident
            # 强行抛出 SystemExit 异常给目标线程
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
            if res == 0:
                self.write_output(">>> 中断失败：无效的线程 ID\n")
            elif res != 1:
                # 恢复异常状态
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                self.write_output(">>> 中断失败：系统错误\n")

    def run_audit_backend(self):
        # 备份原本的终端输出
        original_stdout = sys.stdout 
        original_stderr = sys.stderr
        
        # 将 t1.py 的所有 print 重定向到 GUI 文本框
        sys.stdout = RedirectText(self.output_queue)
        sys.stderr = sys.stdout 

        try:
            company_name = self.company.get()
            pdf_str = self.pdf_entry.get()
            peer_pdf_str = self.peer_pdf_entry.get()
            model = self.model.get()
            tc_api_key = self.tc_api_key.get()
            
            pdf_paths = [Path(p) for p in pdf_str.split(",")] if pdf_str else []
            peer_pdf_paths = [Path(p) for p in peer_pdf_str.split(",")] if peer_pdf_str else []

            ai_service = t1.LLMService(self.api_url.get(), self.api_key.get()) if self.api_key.get() else None
            data_agent = t1.DataAgent(llm_service=ai_service, model=model)
            analysis_agent = t1.AnalysisAgent(llm_service=ai_service, model=model)
            risk_agent = t1.RiskAgent(llm_service=ai_service, model=model)
            audit_agent = t1.AuditAgent(llm_service=ai_service, model=model, tc_api_key=tc_api_key)
            benchmarker = t1.IndustryBenchmarker(data_agent=data_agent, analysis_agent=analysis_agent)
            
            pipeline = t1.AuditPipeline(data_agent, analysis_agent, risk_agent, audit_agent, benchmarker)

            print(f">>> 开始对 {company_name} 进行审计流程...\n")
            pipeline.run(company=company_name, pdf_paths=pdf_paths, peer_pdfs=peer_pdf_paths, api_url=None, output_dir=None)
            print("\n>>> ✅ 审计流程全部完成！")

        except SystemExit:
            # 捕捉我们从主界面用 ctypes 抛过来的退出异常
            print("\n>>> ⚠️ 进程已手动中断！")
        except Exception as e:
            print(f"\n[❌ 系统错误]: {str(e)}")
        finally:
            # 无论如何，一定要把 stdout 恢复，释放按钮状态
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.output_queue.put("DONE_SIGNAL")

    def reset_ui(self):
        self.is_running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

if __name__ == "__main__":
    app = AuditGUI()
    app.mainloop()

    # pyinstaller --noconfirm --onefile --windowed --collect-all customtkinter --collect-all pdfplumber gui_runner.py