import React from 'react';
import 'bootstrap/dist/css/bootstrap.css'
function FileInput (){
    return(
        <div>
            <div className="input-group mb-3">
                <input type="file" className="form-control" id="inputGroupFile02"/>
                <label clclassNameass="input-group-text" for="inputGroupFile02">Upload</label>
            </div>
        </div>
      
    );
}


export default FileInput;