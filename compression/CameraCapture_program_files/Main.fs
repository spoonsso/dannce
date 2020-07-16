module Main

open System
open System.Windows
open System.Xml.Linq
let xn = XName.op_Implicit

open VideoAgent

[<STAThread>]
[<EntryPoint>]
let main argv = 
    let x = XElement.Load(argv.[0])
    let cbs = x.Elements() |> Seq.toList |> List.map (fun x ->
        let fstr s = x.Attribute(xn s).Value
        let fint s = x.Attribute(xn s) |> int
        let ffloat s= x.Attribute(xn s) |> float
        let saveparams cspace = {
            Path = fstr "SavePath" ; 
            NumFramesPerFile=fint "NumFramesPerFile"; 
            FPS=ffloat "FPS" ;Width=fint "Width"; Height=fint "Height";
            ColorSpace=cspace
        }
        let cb = 
            match x.Name.LocalName with
            | "ICubeCam" -> 
                ICubeCam.StartCam,ICubeCam.ColorSpace
            | "WebCam" ->
                WebCam.StartCam,WebCam.ColorSpace
            | "PGRFlea3" ->
                PGRFlyCap.StartCam,PGRFlyCap.ColorSpace
            | _ -> failwith "Camera Not Supported"
            |> fun (f,csp) ->  
                let sp = saveparams csp
                f (VideoAgent sp) sp.Width sp.Height x
        cb
    )
    let app = new Application()
    app.Run()

