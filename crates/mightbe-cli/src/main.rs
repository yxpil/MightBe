//! `mightbe-cli`：交互式 shell 与一次性执行入口（README 4.7）。

use clap::Parser;
use mightbe_cli::{render, wire};
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "mightbe-cli", version, about = "MightBe 命令行客户端")]
struct Args {
    /// 服务器地址
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// 服务器端口
    #[arg(short = 'P', long, default_value_t = 9527)]
    port: u16,

    /// 一次性执行一条语句后退出
    #[arg(short = 'e', long)]
    exec: Option<String>,

    /// 显示每条语句的本地往返耗时
    #[arg(long)]
    timer: bool,
}

fn main() {
    let args = Args::parse();
    let addr = format!("{}:{}", args.host, args.port);
    let stream = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("无法连接 {addr}: {e}");
            std::process::exit(1);
        }
    };
    let mut socket = Socket {
        reader: BufReader::new(stream.try_clone().expect("clone socket")),
        writer: stream,
    };
    let mut session = Session { timer: args.timer };

    match args.exec {
        Some(stmt) => {
            if let Err(e) = session.execute(&mut socket, &stmt) {
                fail(e);
            }
        }
        None => {
            println!("MightBe cli → {addr}（`.help` 查看元命令，`.quit` 退出）");
            if let Err(e) = session.repl(&mut socket, BufReader::new(std::io::stdin())) {
                fail(e);
            }
        }
    }
}

fn fail(e: io::Error) -> ! {
    eprintln!("通信失败: {e}");
    std::process::exit(1);
}

/// 一条连接的两半：读需要 `BufRead`，写需要 `Write`。
struct Socket {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

struct Session {
    timer: bool,
}

impl Session {
    /// 发送语句并渲染响应；`timer` 开启时额外报告本地往返耗时。
    fn execute(&self, sock: &mut Socket, stmt: &str) -> io::Result<()> {
        let started = Instant::now();
        wire::write_request(&mut sock.writer, stmt)?;
        match wire::read_reply(&mut sock.reader)? {
            Some(reply) => {
                print!("{}", render::render(&reply));
                if self.timer {
                    println!("本地耗时 {} ms", started.elapsed().as_millis());
                }
                Ok(())
            }
            None => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "服务器关闭了连接",
            )),
        }
    }

    /// 交互循环：多行语句累积到 `;` 再发送，`.` 开头为元命令。
    fn repl(
        &mut self,
        sock: &mut Socket,
        mut stdin: impl BufRead,
    ) -> io::Result<()> {
        let mut pending = String::new();
        let mut line = String::new();
        loop {
            print!("{}", if pending.is_empty() { "mb> " } else { "   > " });
            io::stdout().flush()?;
            line.clear();
            if stdin.read_line(&mut line)? == 0 {
                println!();
                return Ok(());
            }
            let input = line.trim();
            if input.is_empty() {
                continue;
            }
            if pending.is_empty() && input.starts_with('.') {
                if let MetaOutcome::Stop = self.meta(input) {
                    return Ok(());
                }
                continue;
            }
            pending.push_str(input);
            pending.push(' ');
            if !wire::statement_complete(&pending) {
                continue;
            }
            let stmt = std::mem::take(&mut pending);
            self.execute(sock, &stmt)?;
        }
    }

    fn meta(&mut self, input: &str) -> MetaOutcome {
        let mut parts = input.splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            ".quit" | ".exit" => MetaOutcome::Stop,
            ".help" => {
                println!(
                    ".timer on|off   显示语句耗时\n.help           本帮助\n.quit / .exit   退出\n其余输入按 SQL 发送到服务器（以 ; 结束）"
                );
                MetaOutcome::Handled
            }
            ".timer" => {
                match arg {
                    "on" => self.timer = true,
                    "off" => self.timer = false,
                    other => println!("未知参数 {other:?}，期望 on|off"),
                }
                MetaOutcome::Handled
            }
            other => {
                println!("未知元命令 {other}，`.help` 查看支持项");
                MetaOutcome::Handled
            }
        }
    }
}

enum MetaOutcome {
    Stop,
    Handled,
}
