use std::time::Duration;
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    time::sleep,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").await.unwrap();

    let mut handles = Vec::new();
    for i in 0..2 {
        let (stream, addr) = listener.accept().await.unwrap();
        println!("Accepted {} from {}", i + 1, addr);
        handles.push(tokio::spawn(async move {
            handle_connection(stream).await;
        }));
    }

    for h in handles {
        let _ = h.await;
    }
    println!("Shutting down.");
}

async fn handle_connection(mut stream: TcpStream) {
    let request_line = {
        let mut buf_reader = BufReader::new(&mut stream);
        let mut line = String::new();
        let _ = buf_reader.read_line(&mut line).await;
        line.trim_end().to_string()
    };

    let (status_line, filename) = match request_line.as_str() {
        "GET / HTTP/1.1" => ("HTTP/1.1 200 OK", "hello.html"),
        "GET /sleep HTTP/1.1" => {
            println!("Sleeping 5s");
            sleep(Duration::from_secs(5)).await;
            ("HTTP/1.1 200 OK", "hello.html")
        }
        _ => ("HTTP/1.1 404 NOT FOUND", "404.html"),
    };

    let contents = match fs::read_to_string(filename).await {
        Ok(c) => c,
        Err(e) => {
            println!("Read error: {}", e);
            "Internal Server Error".to_string()
        }
    };
    let length = contents.len();

    let response = format!("{status_line}\r\nContent-Length: {length}\r\n\r\n{contents}");
    let _ = stream.write_all(response.as_bytes()).await;
}
