import { jspModule } from '../generated/jsp.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(jspModule);

export { jspModule };
