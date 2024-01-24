import React from 'react';
import 'bootstrap/dist/css/bootstrap.css'
function TextInputWithLabel (props) {
    return(
        <div>
            <div className="mb-3">
                <label htmlFor="exampleFormControlTextarea1" className="form-label">The Answer</label>
                <textarea className="form-control" id="exampleFormControlTextarea1" rows="7"  value={props.value.answer}/>
            </div>
        </div>
      
    );
}


export default TextInputWithLabel;