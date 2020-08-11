module VideoAgent

open System
open System.IO
open System.Diagnostics
open System.Windows
open System.Windows.Controls
open System.Windows.Media

type Queue<'a> = System.Collections.Generic.Queue<'a>
type Agent<'a> = MailboxProcessor<'a>
type SaveParams = {
    Path : string
    NumFramesPerFile : int
    ColorSpace : string
    Width : int
    Height : int
    FPS : float
}

let VideoAgent (saveparams:SaveParams) =
    let fnum = DateTime.Now.Ticks
    let savepath = sprintf @"%s\%d" saveparams.Path fnum
    Directory.CreateDirectory(savepath)|>ignore
    let timesfile = new BinaryWriter(new FileStream(sprintf @"%s\%d.times" savepath fnum,FileMode.Create))
    let metafile = new FileStream(sprintf @"%s\%d.meta" savepath fnum,FileMode.Create)
    let sw = Stopwatch()
    let qtime = Queue<int64>()
    let nq = 80
    qtime.Enqueue(0L)
    sw.Start()
    Agent.Start(fun inbox ->
        let rec loop framenum (stdin:Stream) = async {
            let! frame,metadata = inbox.Receive()
            timesfile.Write(sw.ElapsedMilliseconds)
            metafile.Write(metadata,0,Array.length metadata)
            let stdin = 
                if framenum%(saveparams.NumFramesPerFile|>int64) = 0L then
                    if stdin<>null then stdin.Close()
                    let p = 
                        let args = 
                            sprintf @"--preset ultrafast --demuxer lavf --input-fmt rawvideo --input-csp %s --input-res %dx%d --fps %.2f -o %s\%d.mkv -" 
                                saveparams.ColorSpace saveparams.Width saveparams.Height saveparams.FPS savepath framenum
                        new Process(StartInfo = ProcessStartInfo("x264",args,
                                        UseShellExecute=false,RedirectStandardInput=true))
                    p.Start()|>ignore
                    p.StandardInput.BaseStream
                else 
                    stdin
            stdin.Write(frame,0,frame.Length)
            return! loop (framenum+1L) stdin
        }
        loop 0L null)

