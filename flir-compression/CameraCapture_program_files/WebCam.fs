module WebCam
open System.Runtime.InteropServices

type callback = delegate of nativeint*int64->unit

let [<Literal>] MFDLL = "MFWebCam.dll"
[<DllImport(MFDLL)>]
extern nativeint StartCapture(uint32 CoInit, [<MarshalAs(UnmanagedType.LPWStr)>]string camSymLink, 
    uint32 streamport, string profilename,callback cb, 
    [<Out>]int64& width, [<Out>]int64& height, [<Out>]System.Guid& subtype);
[<DllImport(MFDLL)>]
extern void StopCapture(nativeint handle)
[<DllImport(MFDLL)>]
extern int GetNumDevices(uint32 CoInit)

open System.Xml.Linq
open System
open System.Threading
open System.Windows.Media.Imaging
open System.Windows.Media
let ig = ignore

let xn = XName.op_Implicit

let StartCam (videoAgent:MailboxProcessor<_>) width height (x:XElement) =
    let symlink = x.Attribute(xn "DevicePath").Value
    let CoInit = if Thread.CurrentThread.GetApartmentState()=ApartmentState.STA then 0x2u else 0x0u
    let cb = callback(fun pBuf time ->
        let frame = Array.zeroCreate<byte> (width*height*2)
        Marshal.Copy(pBuf,frame,0,frame.Length)
        videoAgent.Post(frame,[||])
    )
    let mutable tmpwidth = 0L
    let mutable tmpheight = 0L
    let mutable tmpguid = Guid.Empty
    let handle = StartCapture(CoInit,symlink,0u,"Profile.prx",cb,&tmpwidth,&tmpheight,&tmpguid)
    cb :> obj

let ColorSpace = "yuyv422"
